use crate::model::ModelProfile;

/// A requested candidate: id plus optional overrides.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateInput {
    pub id: String,
    pub quality: Option<f64>,
    pub cost: Option<f64>,
}

/// Built-in model table (seed values; approximate, see SPEC §6).
pub fn builtin() -> Vec<ModelProfile> {
    vec![
        ModelProfile {
            id: "claude-opus-4-8".into(),
            quality: 0.97,
            cost: 0.90,
        },
        ModelProfile {
            id: "claude-sonnet-4-6".into(),
            quality: 0.90,
            cost: 0.45,
        },
        ModelProfile {
            id: "claude-haiku-4-5".into(),
            quality: 0.75,
            cost: 0.12,
        },
        ModelProfile {
            id: "gpt-4o".into(),
            quality: 0.88,
            cost: 0.50,
        },
        ModelProfile {
            id: "gpt-4o-mini".into(),
            quality: 0.62,
            cost: 0.10,
        },
        ModelProfile {
            id: "gemini-1.5-pro".into(),
            quality: 0.85,
            cost: 0.40,
        },
    ]
}

fn lookup(id: &str) -> Option<ModelProfile> {
    builtin().into_iter().find(|m| m.id == id)
}

/// Resolve candidates against the builtin table + overrides.
/// `Err` carries the list of ids that could not be resolved.
pub fn resolve(candidates: &[CandidateInput]) -> Result<Vec<ModelProfile>, Vec<String>> {
    let mut resolved: Vec<ModelProfile> = Vec::new();
    let mut unknown: Vec<String> = Vec::new();

    for c in candidates {
        let profile = match (c.quality, c.cost) {
            (Some(q), Some(co)) => Some(ModelProfile {
                id: c.id.clone(),
                quality: q,
                cost: co,
            }),
            _ => lookup(&c.id).map(|mut base| {
                if let Some(q) = c.quality {
                    base.quality = q;
                }
                if let Some(co) = c.cost {
                    base.cost = co;
                }
                base
            }),
        };

        match profile {
            Some(p) => match resolved.iter_mut().find(|m| m.id == p.id) {
                Some(existing) => *existing = p, // dedup: keep last
                None => resolved.push(p),
            },
            None => {
                if !unknown.contains(&c.id) {
                    unknown.push(c.id.clone());
                }
            }
        }
    }

    if unknown.is_empty() {
        Ok(resolved)
    } else {
        Err(unknown)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(id: &str, quality: Option<f64>, cost: Option<f64>) -> CandidateInput {
        CandidateInput {
            id: id.into(),
            quality,
            cost,
        }
    }

    #[test]
    fn known_id_resolves_to_builtin() {
        let got = resolve(&[input("claude-haiku-4-5", None, None)]).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].quality, 0.75);
        assert_eq!(got[0].cost, 0.12);
    }

    #[test]
    fn full_override_accepts_unknown_id() {
        let got = resolve(&[input("brand-new", Some(0.5), Some(0.2))]).unwrap();
        assert_eq!(
            got[0],
            ModelProfile {
                id: "brand-new".into(),
                quality: 0.5,
                cost: 0.2
            }
        );
    }

    #[test]
    fn partial_override_on_known_merges() {
        let got = resolve(&[input("gpt-4o-mini", None, Some(0.05))]).unwrap();
        assert_eq!(got[0].quality, 0.62); // builtin
        assert_eq!(got[0].cost, 0.05); // overridden
    }

    #[test]
    fn unknown_without_full_override_errors() {
        let err = resolve(&[input("nope", None, None)]).unwrap_err();
        assert_eq!(err, vec!["nope".to_string()]);

        let err2 = resolve(&[input("nope", Some(0.5), None)]).unwrap_err();
        assert_eq!(err2, vec!["nope".to_string()]);
    }

    #[test]
    fn duplicate_ids_keep_last() {
        let got = resolve(&[
            input("gpt-4o-mini", None, None),
            input("gpt-4o-mini", None, Some(0.99)),
        ])
        .unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].cost, 0.99);
    }
}
