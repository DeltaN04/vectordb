//! Payload filters.

#![deny(unsafe_code)]

use serde_json::Value;

#[derive(Debug, Clone)]
pub enum PayloadFilter {
    Eq(String, Value),
    Range(String, Option<f64>, Option<f64>), // field, min, max inclusive
    In(String, Vec<Value>),
    And(Vec<PayloadFilter>),
    Or(Vec<PayloadFilter>),
    Not(Box<PayloadFilter>),
}

pub trait Filter {
    fn eval(&self, payload: Option<&Value>) -> bool;
}

impl Filter for PayloadFilter {
    fn eval(&self, payload: Option<&Value>) -> bool {
        let p = match payload {
            Some(v) => v,
            None => return false,
        };
        match self {
            PayloadFilter::Eq(field, val) => p.get(field).map(|v| v == val).unwrap_or(false),
            PayloadFilter::Range(field, min, max) => {
                if let Some(n) = p.get(field).and_then(|v| v.as_f64()) {
                    if let Some(mn) = min {
                        if n < *mn {
                            return false;
                        }
                    }
                    if let Some(mx) = max {
                        if n > *mx {
                            return false;
                        }
                    }
                    true
                } else {
                    false
                }
            }
            PayloadFilter::In(field, vals) => {
                if let Some(v) = p.get(field) {
                    vals.iter().any(|x| x == v)
                } else {
                    false
                }
            }
            PayloadFilter::And(fs) => fs.iter().all(|f| f.eval(payload)),
            PayloadFilter::Or(fs) => fs.iter().any(|f| f.eval(payload)),
            PayloadFilter::Not(f) => !f.eval(payload),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn eq_filter() {
        let p = json!({"color":"red","n":5});
        let f = PayloadFilter::Eq("color".into(), json!("red"));
        assert!(f.eval(Some(&p)));
        let f2 = PayloadFilter::Eq("color".into(), json!("blue"));
        assert!(!f2.eval(Some(&p)));
    }

    #[test]
    fn range_and_not() {
        let p = json!({"n":5});
        let f = PayloadFilter::Range("n".into(), Some(3.0), Some(10.0));
        assert!(f.eval(Some(&p)));
        let f2 = PayloadFilter::Not(Box::new(PayloadFilter::Eq("color".into(), json!("red"))));
        assert!(f2.eval(Some(&p)));
    }

    #[test]
    fn and_or_in() {
        let p = json!({"color":"red","tags":["a","b"]});
        let f = PayloadFilter::And(vec![PayloadFilter::In(
            "color".into(),
            vec![json!("red"), json!("blue")],
        )]);
        assert!(f.eval(Some(&p)));
        let f2 = PayloadFilter::Or(vec![
            PayloadFilter::Eq("color".into(), json!("blue")),
            PayloadFilter::Eq("color".into(), json!("red")),
        ]);
        assert!(f2.eval(Some(&p)));
    }
}
