#[derive(Debug, Clone, PartialEq)]
pub struct EngineConfig {
    pub board_width: usize,
    pub visible_height: usize,
    pub buffer_height: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            board_width: 10,
            visible_height: 20,
            buffer_height: 20,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct InputFrame {
    pub dt_seconds: f32,
    pub left: bool,
    pub right: bool,
    pub soft_drop: bool,
    pub hard_drop: bool,
    pub rotate_clockwise: bool,
    pub rotate_counterclockwise: bool,
    pub hold: bool,
    pub pause: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineEvent {}

#[derive(Debug, Clone, PartialEq)]
pub struct EngineSnapshot {
    pub config: EngineConfig,
}

#[derive(Debug, Clone)]
pub struct Engine {
    config: EngineConfig,
}

impl Engine {
    pub fn new(config: EngineConfig) -> Self {
        Self { config }
    }

    pub fn step(&mut self, _input: InputFrame) -> Vec<EngineEvent> {
        Vec::new()
    }

    pub fn snapshot(&self) -> EngineSnapshot {
        EngineSnapshot {
            config: self.config.clone(),
        }
    }
}
