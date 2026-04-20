/// Configuration for catapult usage in flat graph searches.
///
/// In flat searches, catapults are either globally enabled or disabled.
#[repr(u8)]
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum CatapultChoice {
    /// Catapults are disabled; only the base starting node is used
    CatapultsDisabled = 0,

    /// Catapults are enabled; cached starting points from similar queries are used
    CatapultsEnabled = 1,
}

impl CatapultChoice {
    /// # Returns
    /// `true` if catapults are enabled, `false` otherwise
    pub fn enabled(self) -> bool {
        self == CatapultChoice::CatapultsEnabled
    }

    /// Creates a `FlatCatapultChoice` from a boolean flag.
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable catapults
    ///
    /// # Returns
    /// `CatapultsEnabled` if `enabled` is true, `CatapultsDisabled` otherwise
    pub fn from_bool(enabled: bool) -> Self {
        if enabled {
            CatapultChoice::CatapultsEnabled
        } else {
            CatapultChoice::CatapultsDisabled
        }
    }
}
