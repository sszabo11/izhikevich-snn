#![allow(non_snake_case)]
use anyhow::Result;
use ndarray::{Array1, Array2};
use snn_2::{network::Network, plt::Plt};

fn main() -> Result<()> {
    let T = 10;
    let input_dim = 16;
    let output_dim = 16;

    let mut network = Network::new(16, 2, input_dim, output_dim, 0.8, 0.1, 0.3, 0.8, T);

    let pattern1: Vec<u32> = [
        0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0,
        1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1,
        1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1,
        0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0,
        0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0,
        1, 0, 0, 1, 0, 1, 1, 0, 1, 0,
    ]
    .to_vec();

    let mut input_train: Array2<u32> = Array2::from_shape_vec((T, input_dim), pattern1).unwrap();

    let mut output_train = Array2::zeros((T, input_dim));

    network.run(&input_train, &mut output_train, T as u32);

    Plt::raster("raster1.png", &input_train)?;

    Ok(())
}
