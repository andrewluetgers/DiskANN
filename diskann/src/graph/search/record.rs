/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::fmt::Display;

use crate::neighbor::Neighbor;

/// How far up the "usefulness ladder" a single traversed out-link got.
///
/// Ordered so that a larger discriminant implies every weaker condition also held: an edge that
/// reached [`EdgeOutcome::ExpandedAsBeam`] necessarily also entered the frontier, passed the filter
/// and was not a duplicate. Each variant is the *terminal* state recorded for that edge.
///
/// The point of separating these rather than recording a single "useful" bool is that they imply
/// different actions: `AlreadyVisited` is pure wasted IO (the adjacency entry was read and then
/// discarded without even a distance computation), whereas `NotCloserThanFrontier` means the edge
/// was genuinely evaluated and simply lost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum EdgeOutcome {
    /// L0 only: the adjacency entry was read but skipped before any further work.
    Read = 0,
    /// L1 failed: the neighbour was already visited, so no distance was computed. Wasted IO.
    AlreadyVisited = 1,
    /// L2 failed: distance computed, but the inline filter rejected the neighbour.
    FilterRejected = 2,
    /// L3 failed: passed the filter but was not close enough to enter the candidate list.
    NotCloserThanFrontier = 3,
    /// L3: entered the candidate list, i.e. it improved the frontier.
    EnteredFrontier = 4,
    /// L4: was later popped as a beam node, i.e. it actually steered the walk.
    ExpandedAsBeam = 5,
}

impl EdgeOutcome {
    /// Number of distinct outcomes, for sizing fixed histogram arrays without a HashMap.
    pub const COUNT: usize = 6;

    /// Discriminant as an index into such an array.
    pub fn as_index(self) -> usize {
        self as usize
    }
}

/// Sentinel `rank` meaning "this edge was not attributed to a parent/position".
///
/// Deliberately `u8::MAX` rather than `0`: rank 0 is a real and highly meaningful value (the
/// nearest neighbour, which is the true nearest neighbour ~97% of the time in the corpus this was
/// built for), so conflating "unknown" with "nearest" would corrupt exactly the per-rank histogram
/// this instrumentation exists to produce. Chosen over `Option<u8>` to keep the hot-path callback
/// signature branch-free.
pub const RANK_UNATTRIBUTED: u8 = u8::MAX;

/// A logger provided to various search tasks
///
/// # Why no `'static` supertrait
/// This originally read `Send + Sync + 'static`. The `'static` was relaxed so a *borrowed*
/// diagnostic sink can be attached to a single search
/// ([`InlineFilterSearch::with_edge_record`](super::InlineFilterSearch::with_edge_record)): with
/// `'static` in the supertrait, `dyn SearchRecord<T>` is implicitly `dyn SearchRecord<T> + 'static`,
/// so `&'r mut dyn SearchRecord<T>` forces `'r: 'static` and a sink holding any borrowed state
/// (e.g. a slice of per-vector routing ids) cannot be passed in at all.
///
/// This is a *relaxation*, so every existing implementor still satisfies it — `NoopSearchRecord`,
/// `VisitedSearchRecord` and `RecallSearchRecord` are all `'static` regardless. Only code that
/// requires `SR: 'static` (e.g. spawning the search future onto an executor rather than awaiting
/// it in place) would notice, and that would surface as a compile error at the call site rather
/// than as a runtime problem.
pub trait SearchRecord<T>: Send + Sync
where
    T: Eq,
{
    /// Provides a customization point for logging done during search.
    ///
    /// # Parameters
    /// - `neighbor`: The neighbor node being recorded.
    /// - `hops`: The total number of hops taken to reach this neighbor.
    /// - `cmps`: The total number of comparisons performed to reach this neighbor.
    ///
    /// # Default Implementation
    /// The default implementation of this method is a noop, as in most contexts logging is not required.
    ///
    /// # Type Parameters
    /// - `T`: The data type associated with the neighbor.
    fn record(&mut self, _neighbor: Neighbor<T>, _hops: u32, _cmps: u32) {
        // Default no-op implementation
    }

    /// Whether this sink wants per-edge records at all.
    ///
    /// Recording edges costs more than the `record_edge` call itself: the traversal has to take a
    /// *traced* beam-expansion path that reports adjacency entries the production path drops
    /// without ever materialising. That is worth paying only for a sink that will use them.
    ///
    /// # Default implementation
    /// `false`. For [`NoopSearchRecord`] this monomorphises to a constant, so the traced branch at
    /// the call site is dead code and eliminated outright — production does not even evaluate a
    /// condition. A diagnostic sink overrides this to `true`.
    fn wants_edges(&self) -> bool {
        false
    }

    /// Records one traversed out-link: which edge it was, and how useful it turned out to be.
    ///
    /// # Parameters
    /// - `parent`: the node whose adjacency list was being expanded.
    /// - `rank`: position within that adjacency list, 0-based, in the stored nearest-first order.
    ///   [`RANK_UNATTRIBUTED`] when the caller could not attribute the edge.
    /// - `parent_out_degree`: how many out-links `parent` actually had. Not a constant — real
    ///   graphs have ragged degree (mean 29.89, min 1, max 32 in the corpus this was written for),
    ///   so the denominator for "what fraction of this node's edges were useful" must be recorded
    ///   per node rather than assumed to be `max_degree`.
    /// - `child`: the neighbour the edge points at.
    /// - `dist`: query-to-`child` distance, or `None` when the edge was discarded before any
    ///   distance was computed. `None` means "not measured", which is not the same as zero.
    /// - `outcome`: the terminal [`EdgeOutcome`] for this edge.
    /// - `hop`: the beam iteration during which the edge was read.
    ///
    /// # Default Implementation
    /// A no-op, so existing implementors are unaffected and [`NoopSearchRecord`] monomorphises
    /// this away entirely rather than paying a branch per edge.
    #[allow(clippy::too_many_arguments)]
    fn record_edge(
        &mut self,
        _parent: T,
        _rank: u8,
        _parent_out_degree: u8,
        _child: T,
        _dist: Option<f32>,
        _outcome: EdgeOutcome,
        _hop: u16,
    ) {
        // Default no-op implementation
    }
}

//////////////////////
// NoopSearchRecord //
//////////////////////

/// A empty struct implementing `SearchRecord`.
///
/// Used for situations where a search record is not needed.
/// This is most common for production code where logging is not required, outside of index building.
#[derive(Default)]
pub struct NoopSearchRecord;

impl Display for NoopSearchRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "noop search record")
    }
}

impl NoopSearchRecord {
    pub fn new() -> Self {
        NoopSearchRecord
    }
}

impl<T> SearchRecord<T> for NoopSearchRecord where T: Eq {}

#[derive(Default)]
pub struct VisitedSearchRecord<T>
where
    T: Eq + Clone + Send + Sync + 'static,
{
    pub visited: Vec<Neighbor<T>>,
}

impl<T> std::fmt::Display for VisitedSearchRecord<T>
where
    T: Eq + Clone + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "visited search record")
    }
}

impl<T> VisitedSearchRecord<T>
where
    T: Eq + Clone + Send + Sync + 'static,
{
    pub fn new(initial_reservation: usize) -> Self {
        Self {
            visited: Vec::with_capacity(initial_reservation),
        }
    }

    pub fn push(&mut self, neighbor: Neighbor<T>) {
        self.visited.push(neighbor);
    }

    pub fn ids(&self) -> impl ExactSizeIterator<Item = T> + Clone + Send + Sync {
        self.visited.iter().map(|n| n.id.clone())
    }
}

impl<T> SearchRecord<T> for VisitedSearchRecord<T>
where
    T: Eq + Clone + Send + Sync + 'static,
{
    fn record(&mut self, neighbor: Neighbor<T>, _hops: u32, _cmps: u32) {
        self.push(neighbor);
    }
}

pub struct RecallSearchRecord<T>
where
    T: Eq + Clone + Send + Sync + 'static,
{
    groundtruth: Vec<T>,
    running_recall: usize,

    pub hops: Vec<u32>,
    pub recall: Vec<usize>,
}

impl<T> std::fmt::Display for RecallSearchRecord<T>
where
    T: Eq + Clone + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "recall search record")
    }
}

impl<T> RecallSearchRecord<T>
where
    T: Eq + Clone + Send + Sync + 'static,
{
    pub fn new(initial_reservation: usize, groundtruth: Vec<T>) -> Self {
        Self {
            groundtruth,
            running_recall: 0,
            hops: Vec::with_capacity(initial_reservation),
            recall: Vec::with_capacity(initial_reservation),
        }
    }

    pub fn push(&mut self, neighbor: Neighbor<T>, hops: u32) {
        self.hops.push(hops);
        if self.groundtruth.contains(&neighbor.id) {
            self.running_recall += 1;
        }
        self.recall.push(self.running_recall);
    }
}

impl<T> SearchRecord<T> for RecallSearchRecord<T>
where
    T: Eq + Clone + Send + Sync + 'static,
{
    fn record(&mut self, neighbor: Neighbor<T>, hops: u32, _cmps: u32) {
        self.push(neighbor, hops);
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    ////////////////////
    // DefaultContext //
    ////////////////////

    #[test]
    fn test_default_record() {
        let record = NoopSearchRecord;

        // Check that the implementation of `Display` is correct.
        assert_eq!(record.to_string(), "noop search record");

        assert_eq!(
            std::mem::size_of::<NoopSearchRecord>(),
            0,
            "expected NoopSearchRecord to be an empty class"
        );
    }

    #[test]
    fn test_default_search_record() {
        let mut record = NoopSearchRecord::new();
        record.record(Neighbor::new(1, 2.0), 2, 3);
    }

    /////////////////////////
    // VisitedSearchRecord //
    /////////////////////////

    #[test]
    fn test_visited_search_record() {
        let record: VisitedSearchRecord<u32> = VisitedSearchRecord::new(1);

        // Check that the implementation of `Display` is correct.
        assert_eq!(record.to_string(), "visited search record");
    }

    #[test]
    fn test_visited_search_record_logging() {
        let mut record = VisitedSearchRecord::new(1);
        record.push(Neighbor::new(4, 5.0));
        record.record(Neighbor::new(1, 2.0), 2, 3);

        assert_eq!(
            record.visited.len(),
            2,
            "Expected two neighbors to be logged"
        );
    }

    ////////////////////////
    // RecallSearchRecord //
    ////////////////////////

    #[test]
    fn test_recall_search_record() {
        let record: RecallSearchRecord<u32> = RecallSearchRecord::new(1, vec![1, 2, 3]);

        // Check that the implementation of `Display` is correct.
        assert_eq!(record.to_string(), "recall search record");
    }

    #[test]
    fn test_recall_search_record_logging() {
        let mut record = RecallSearchRecord::new(1, vec![1, 2, 3]);
        record.record(Neighbor::new(4, 5.0), 1, 4);
        record.record(Neighbor::new(1, 2.0), 2, 3);

        assert_eq!(record.hops.len(), 2, "Expected two hop entries");
        assert_eq!(record.recall.len(), 2, "Expected two recall entries");

        assert_eq!(record.recall[0], 0, "Expected first recall to be 0");
        assert_eq!(record.recall[1], 1, "Expected second recall to be 1");
    }
}
