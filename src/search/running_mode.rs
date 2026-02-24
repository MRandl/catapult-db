#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunningMode {
    Vanilla,
    Catapult,
    LshApg,
}

impl RunningMode {
    pub fn from_string(s: &str) -> Self {
        if s == "vanilla" {
            RunningMode::Vanilla
        } else if s == "catapult" {
            RunningMode::Catapult
        } else if s == "lshapg" {
            RunningMode::LshApg
        } else {
            panic!("Invalid runtime mode: {}", s)
        }
    }
}
