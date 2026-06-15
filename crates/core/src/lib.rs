//! route-llm core: pure routing logic (no I/O).

pub mod budget;
pub mod difficulty;
pub mod learned;
pub mod model;
pub mod ranker;
pub mod registry;
pub mod router;

pub use budget::BudgetRouter;
pub use learned::LearnedRouter;
pub use model::{
    BudgetBreakdown, Difficulty, DimensionScores, ModelProfile, Policy, RankedModel,
    Recommendation, RoutingPreferences,
};
pub use registry::CandidateInput;
pub use router::{HeuristicRouter, Router};
