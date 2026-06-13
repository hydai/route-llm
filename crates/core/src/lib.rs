//! route-llm core: pure routing logic (no I/O).

pub mod difficulty;
pub mod model;

pub use model::{Difficulty, ModelProfile, RankedModel, Recommendation, RoutingPreferences};
