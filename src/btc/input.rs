use ndarray::Array1;

use crate::btc::csv::BtcRecord;

pub fn encode_btc(data: &Vec<BtcRecord>) -> Array1<f32> {
    let mut spikes = Array1::zeros(data.len() - 1);
    for i in 0..data.len() {
        if i == 0 {
            continue;
        };

        let curr = &data[i];
        let prev = &data[i - 1];

        if curr.close >= prev.close {
            spikes[i] = 1.;
        }
    }

    spikes
}
