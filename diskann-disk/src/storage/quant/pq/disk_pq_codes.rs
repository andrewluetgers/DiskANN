/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Disk-resident PQ codes ("scale mode").
//!
//! A resident index loads every PQ code (`num_points × num_pq_chunks` bytes) into RAM at open.
//! At 250M points × 192 chunks that is ~48 GB per copy, so scale mode keeps only the pivot table
//! in RAM and reads the codes a search needs straight out of `<prefix>_pq_compressed.bin`.
//!
//! File layout (unchanged from the resident format): `[u32 n][u32 num_chunks]` then `n ×
//! num_chunks` code bytes in id order, so the code for `id` starts at byte `8 + id * num_chunks`.
//! A code may straddle a page boundary.
//!
//! One [`PQCodeReader`] lives in each search scratch (one per concurrent search), mirroring the
//! per-scratch vertex provider and its sector reader. A gather of `ids` sorts and dedupes the
//! [`PQ_CODE_PAGE_SIZE`] pages covering their codes, reads them in one batched call (one aligned
//! read per distinct page), then copies each id's code out in REQUEST order. The caller scores the
//! gathered codes with the same `compute_pq_distance` kernel the resident path uses, so distances
//! are bit-identical to resident mode.

use std::{fmt, sync::Arc};

use diskann::{ANNError, ANNResult};
use diskann_quantization::{
    alloc::{AlignedAllocator, Poly},
    num::PowerOfTwo,
};

use crate::utils::aligned_file_reader::{
    traits::{AlignedFileReader, AlignedReaderFactory},
    AlignedRead, Alignment,
};

/// Granularity of a code-page read. 4 KiB matches the node sector size and the device page, and is
/// a multiple of every reader's alignment (512 for O_DIRECT, 1 for the buffered fallback).
pub const PQ_CODE_PAGE_SIZE: usize = 4096;

/// `[u32 num_points][u32 num_chunks]` precedes the codes.
const PQ_CODES_HEADER_BYTES: u64 = 8;

/// Where an index's PQ codes live while it is being searched. Chosen at open; does not change the
/// on-disk format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PQResidency {
    /// Load every code into RAM at open (the original behaviour).
    #[default]
    Resident,
    /// Keep codes in the file; only the pivot table is held in RAM. Each PQ distance batch reads
    /// the pages holding its codes.
    Disk,
}

/// Object-safe view of an [`AlignedFileReader`] that reads whole code pages, so `PQData` (and the
/// search scratch) need not be generic over the reader type.
trait CodePageRead: Send + Sync {
    /// Fill `buf` (`pages.len() × PQ_CODE_PAGE_SIZE` bytes, page-aligned) with the given pages,
    /// in order. Bytes of a page past end-of-file are left unspecified.
    fn read_pages(&mut self, pages: &[u64], file_len: u64, buf: &mut [u8]) -> ANNResult<()>;
}

impl<R: AlignedFileReader> CodePageRead for R {
    fn read_pages(&mut self, pages: &[u64], file_len: u64, buf: &mut [u8]) -> ANNResult<()> {
        let align = <R::Alignment as Alignment>::VALUE.raw();
        let mut requests = Vec::with_capacity(pages.len());
        for (page, slot) in pages.iter().zip(buf.chunks_exact_mut(PQ_CODE_PAGE_SIZE)) {
            let offset = page * PQ_CODE_PAGE_SIZE as u64;
            // The final page is usually partial. Trim the request to the reader's alignment rather
            // than a full page: an O_DIRECT read past EOF just comes back short, while the
            // buffered fallback (alignment 1) would index out of bounds on a full page.
            let remaining = file_len.saturating_sub(offset) as usize;
            let len = remaining.min(PQ_CODE_PAGE_SIZE).next_multiple_of(align);
            requests.push(AlignedRead::<u8, R::Alignment>::new(offset, &mut slot[..len])?);
        }
        self.read(&mut requests)
    }
}

/// Object-safe view of an [`AlignedReaderFactory`]; see [`CodePageRead`].
trait CodePageReaderFactory: Send + Sync {
    fn build(&self) -> ANNResult<Box<dyn CodePageRead>>;
}

impl<F> CodePageReaderFactory for F
where
    F: AlignedReaderFactory,
    F::AlignedReaderType: 'static,
{
    fn build(&self) -> ANNResult<Box<dyn CodePageRead>> {
        Ok(Box::new(AlignedReaderFactory::build(self)?))
    }
}

/// The disk-resident code table: its shape plus a factory for per-search readers over the codes
/// file. Held by `PQData` in place of the in-RAM code matrix.
#[derive(Clone)]
pub struct DiskPQCodes {
    num_points: usize,
    num_chunks: usize,
    file_len: u64,
    factory: Arc<dyn CodePageReaderFactory>,
}

impl fmt::Debug for DiskPQCodes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DiskPQCodes")
            .field("num_points", &self.num_points)
            .field("num_chunks", &self.num_chunks)
            .field("file_len", &self.file_len)
            .finish_non_exhaustive()
    }
}

impl DiskPQCodes {
    /// `factory` opens the `_pq_compressed.bin` file; `file_len` is its length in bytes.
    pub fn new<F>(
        factory: F,
        num_points: usize,
        num_chunks: usize,
        file_len: u64,
    ) -> ANNResult<Self>
    where
        F: AlignedReaderFactory + 'static,
        F::AlignedReaderType: 'static,
    {
        if num_chunks == 0 {
            return Err(ANNError::log_index_error("pq codes file has zero chunks"));
        }
        let needed = PQ_CODES_HEADER_BYTES + (num_points * num_chunks) as u64;
        if file_len < needed {
            return Err(ANNError::log_index_error(format!(
                "pq codes file is {file_len} bytes, too short for {num_points} x {num_chunks} codes"
            )));
        }
        Ok(Self {
            num_points,
            num_chunks,
            file_len,
            factory: Arc::new(factory),
        })
    }

    pub fn num_points(&self) -> usize {
        self.num_points
    }

    /// A fresh reader over the codes file, for one search scratch.
    pub(crate) fn reader(&self) -> ANNResult<PQCodeReader> {
        Ok(PQCodeReader {
            reader: self.factory.build()?,
            num_points: self.num_points,
            num_chunks: self.num_chunks,
            file_len: self.file_len,
            pages: Vec::new(),
            page_buf: None,
            gathered: Vec::new(),
            positions: Vec::new(),
        })
    }
}

/// Per-search reader and scratch for disk-resident PQ codes. Reused across searches through the
/// scratch pool, so its buffers grow to the largest batch seen and then stop allocating.
pub(crate) struct PQCodeReader {
    reader: Box<dyn CodePageRead>,
    num_points: usize,
    num_chunks: usize,
    file_len: u64,
    /// Distinct pages of the current batch, ascending.
    pages: Vec<u64>,
    /// Page-aligned landing buffer, `pages.len() × PQ_CODE_PAGE_SIZE` bytes or more.
    page_buf: Option<Poly<[u8], AlignedAllocator>>,
    /// Gathered codes, `ids.len() × num_chunks`, in request order.
    gathered: Vec<u8>,
    /// `0, 1, 2, …`: the ids to hand `compute_pq_distance` so it scores `gathered` row by row.
    positions: Vec<u32>,
}

/// The result of [`PQCodeReader::gather`].
pub(crate) struct GatheredCodes<'a> {
    /// `rows.len() × num_chunks` code bytes, in request order.
    pub codes: &'a [u8],
    /// `0..n`, the ids that address `codes` as a code table.
    pub rows: &'a [u32],
    /// Distinct code pages read for this batch.
    pub pages_read: usize,
}

/// Byte offset of `id`'s code in the codes file.
fn code_offset(id: u32, num_chunks: usize) -> u64 {
    PQ_CODES_HEADER_BYTES + id as u64 * num_chunks as u64
}

impl PQCodeReader {
    /// Read the codes of `ids` and return them packed contiguously in request order
    /// (`ids.len() × num_chunks` bytes), the row ids `0..ids.len()` addressing them, and the number
    /// of pages read.
    pub(crate) fn gather(&mut self, ids: &[u32]) -> ANNResult<GatheredCodes<'_>> {
        let chunks = self.num_chunks as u64;
        let page = PQ_CODE_PAGE_SIZE as u64;

        self.pages.clear();
        self.gathered.clear();
        if ids.is_empty() {
            return Ok(GatheredCodes {
                codes: &self.gathered,
                rows: &[],
                pages_read: 0,
            });
        }
        for &id in ids {
            if id as usize >= self.num_points {
                return Err(ANNError::log_index_error(format!(
                    "pq code for id {id} requested, but the codes file holds {} points",
                    self.num_points
                )));
            }
            let start = code_offset(id, self.num_chunks);
            // Every page the code touches: a code straddling a boundary needs both.
            self.pages.extend(start / page..=(start + chunks - 1) / page);
        }
        self.pages.sort_unstable();
        self.pages.dedup();

        let needed = self.pages.len() * PQ_CODE_PAGE_SIZE;
        // Grow-only, so a pooled reader stops allocating once it has seen its largest batch.
        let buf = match self.page_buf.take() {
            Some(buf) if buf.len() >= needed => buf,
            _ => {
                let alloc = AlignedAllocator::new(
                    PowerOfTwo::new(PQ_CODE_PAGE_SIZE).map_err(ANNError::log_index_error)?,
                );
                Poly::broadcast(0u8, needed, alloc).map_err(ANNError::log_index_error)?
            }
        };
        let buf = self.page_buf.insert(buf);
        self.reader
            .read_pages(&self.pages, self.file_len, &mut buf[..needed])?;

        for &id in ids {
            let start = code_offset(id, self.num_chunks);
            // `pages` is sorted and deduped and holds every page of this code, so a straddling
            // code's second page sits directly after its first in `buf`: the code is contiguous.
            let slot = self.pages.partition_point(|&p| p < start / page);
            let at = slot * PQ_CODE_PAGE_SIZE + (start % page) as usize;
            self.gathered
                .extend_from_slice(&buf[at..at + self.num_chunks]);
        }
        let n = ids.len() as u32;
        if (self.positions.len() as u32) < n {
            self.positions.extend(self.positions.len() as u32..n);
        }
        Ok(GatheredCodes {
            codes: &self.gathered,
            rows: &self.positions[..ids.len()],
            pages_read: self.pages.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::VirtualAlignedReaderFactory;
    use diskann_providers::storage::{StorageWriteProvider, VirtualStorageProvider};
    use std::io::Write;

    /// Codes spanning several pages, read back through the buffered reader, must match the source
    /// bytes exactly and in request order, including codes that straddle a page boundary and a
    /// final partial page.
    #[test]
    fn gather_returns_codes_in_request_order() {
        let num_points = 300usize;
        let num_chunks = 37usize; // odd so codes straddle 4 KiB boundaries
        let mut file = Vec::new();
        file.extend_from_slice(&(num_points as u32).to_le_bytes());
        file.extend_from_slice(&(num_chunks as u32).to_le_bytes());
        for i in 0..num_points * num_chunks {
            file.push((i * 31 % 251) as u8);
        }
        let storage = Arc::new(VirtualStorageProvider::new_memory());
        storage
            .create_for_write("/codes.bin")
            .unwrap()
            .write_all(&file)
            .unwrap();

        let codes = DiskPQCodes::new(
            VirtualAlignedReaderFactory::new("/codes.bin".to_string(), Arc::clone(&storage)),
            num_points,
            num_chunks,
            file.len() as u64,
        )
        .unwrap();
        let mut reader = codes.reader().unwrap();

        let ids = [299u32, 0, 110, 111, 110, 5];
        let got = reader.gather(&ids).unwrap();
        assert!(got.pages_read >= 2);
        assert_eq!(got.rows, &[0, 1, 2, 3, 4, 5]);
        let gathered = got.codes;
        for (k, &id) in ids.iter().enumerate() {
            let at = 8 + id as usize * num_chunks;
            assert_eq!(
                &gathered[k * num_chunks..(k + 1) * num_chunks],
                &file[at..at + num_chunks],
                "code for id {id} at position {k}"
            );
        }
        assert!(reader.gather(&[num_points as u32]).is_err());
        let empty = reader.gather(&[]).unwrap();
        assert_eq!((empty.codes.len(), empty.rows.len(), empty.pages_read), (0, 0, 0));
    }
}
