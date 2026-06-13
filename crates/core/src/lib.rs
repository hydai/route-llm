//! route-llm core: pure routing logic (no I/O).

pub mod difficulty;
pub mod model;
pub mod ranker;
pub mod registry;

pub use model::{Difficulty, ModelProfile, RankedModel, Recommendation, RoutingPreferences};
pub use registry::CandidateInput;
