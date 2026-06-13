//! route-llm core: pure routing logic (no I/O).

pub mod difficulty;
pub mod learned;
pub mod model;
pub mod ranker;
pub mod registry;
pub mod router;

pub use model::{Difficulty, ModelProfile, RankedModel, Recommendation, RoutingPreferences};
pub use registry::CandidateInput;
pub use router::{HeuristicRouter, Router};
