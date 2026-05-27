use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct BtcRecord {
    #[serde(rename = "Timestamp")]
    pub timestamp: f32,
    #[serde(rename = "Open")]
    pub open: f32,
    #[serde(rename = "High")]
    pub high: f32,
    #[serde(rename = "Close")]
    pub close: f32,
    #[serde(rename = "Low")]
    pub low: f32,
    #[serde(rename = "Volume")]
    pub volume: f32,
}

pub fn parse_csv(path: &str, from_timestamp: u32) -> Result<Vec<BtcRecord>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;

    let mut data = Vec::new();

    for result in reader.deserialize() {
        let record: BtcRecord = result?;
        if record.timestamp as u32 >= from_timestamp {
            data.push(record);
        }
    }

    Ok(data)
}
