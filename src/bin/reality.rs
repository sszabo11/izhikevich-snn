use anyhow::Result;
use ndarray::{Array1, Array2, ArrayView1, array};
use snn_2::{
    evolve::{Reality, check_xor},
    layer::{IzhikevichLayerBuilder, IzhikevichModel, Logger},
    plot::{
        plot_all_layers, plot_all_layers_raster, plot_izhikievich, plot_raster, plot_weights,
        plot_weights2,
    },
};

fn main() -> Result<()> {
    const T: usize = 10;

    let data: Array2<f32> = array![
        [1., 1., 1., -1., -1., -1., -1., -1., -1.], // your original
        [1., -1., 1., 1., -1., 1., 1., -1., 1.],    // all +
        [-1., 1., -1., -1., 1., 1., -1., -1., -1.], // all -
        [1., -1., 1., -1., 1., -1., 1., -1., 1.],   // checkerboard
        [-1., 1., -1., 1., -1., 1., -1., 1., -1.],  // inverted checkerboard
        [1., 1., -1., 1., 1., -1., 1., 1., -1.],    // vertical stripes
        [1., -1., 1., 1., -1., 1., 1., -1., 1.],    // horizontal stripes
        [1., 1., 1., -1., -1., -1., 1., 1., 1.],    // top+bottom
        [-1., -1., -1., 1., 1., 1., -1., -1., -1.], // middle bar
        [1., -1., -1., -1., 1., -1., -1., -1., 1.]  // X-like
    ];
    //let target: Array2<f32> = array![
    //    [1., 1., 1., -1., -1., -1., -1., -1., -1.], // same as input (identity)
    //    [1., 1., 1., 1., 1., 1., 1., 1., 1.],
    //    [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
    //    [-1., 1., -1., 1., -1., 1., -1., 1., -1.], // inverted checkerboard
    //    [1., -1., 1., -1., 1., -1., 1., -1., 1.],  // checkerboard
    //    [-1., -1., 1., -1., -1., 1., -1., -1., 1.],
    //    [-1., 1., -1., -1., 1., -1., -1., 1., -1.],
    //    [-1., -1., -1., 1., 1., 1., -1., -1., -1.],
    //    [1., 1., 1., -1., -1., -1., 1., 1., 1.],
    //    [-1., 1., 1., 1., -1., 1., 1., 1., -1.]
    //];
    let target: Array2<f32> = array![
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.], // for original
        [-1., -1., -1., 1., 1., 1., 1., 1., 1.],       // for all +
        [1., 1., 1., -1., -1., -1., -1., -1., -1.],    // for all -
        [-1., 1., -1., -1., 1., -1., 1., -1., 1.],     // for checkerboard
        [1., -1., 1., 1., -1., 1., -1., 1., -1.],      // for inverted checkerboard
        [-1., -1., 1., 1., 1., -1., 1., 1., -1.],      // for vertical stripes
        [-1., 1., -1., 1., -1., 1., 1., -1., 1.],      // for horizontal stripes
        [-1., -1., -1., -1., -1., -1., 1., 1., 1.],    // for top+bottom
        [1., 1., 1., 1., 1., 1., -1., -1., -1.],       // for middle bar
        [-1., 1., 1., -1., 1., -1., -1., -1., 1.]      // for X-like
    ];

    let mut reality = Reality::new(T);

    reality.adam_and_eve();

    //while reality.step < 200 {
    while reality.running && reality.step < 15000 {
        let i = (reality.step % 10) as usize;
        reality.tick(data.row(i), target.row(i));

        if reality.step % 1 == 0 {
            println!(
                "Generation: {}. Living: {}",
                reality.step,
                reality.brains.len()
            );
        }
    }

    for brain in reality.brains.iter_mut() {
        let input = data.row(0);
        let output = brain.model.run(T, input);
        println!("input: {}", input);
        println!("output: {}", output);

        let corr = check_res(input.view(), output.view());
        println!("{}: Accuracy: {}%", brain.id, corr / 9. * 100.);
        println!()
    }

    plot_all_layers(&reality.brains[0].model.layers, "charts/layerd-2.png").expect("fasield to ");
    plot_all_layers_raster(&reality.brains[0].model.layers, "charts/l_rast-2.png")
        .expect("fasield to ");

    Ok(())
}

//pub fn check_res(input: ArrayView1<f32>, output: ArrayView1<f32>) -> f32 {
//    let mut wrong: f32 = 0.;
//    for (i, &v) in input.iter().enumerate() {
//        let d = if v == -1. { false } else { true };
//        let o = if output[i] == 0. { false } else { true };
//
//        if o == d {
//            wrong += 1.;
//        }
//    }
//
//    wrong
//}

pub fn check_res(input: ArrayView1<f32>, output: ArrayView1<f32>) -> f32 {
    let mut correct: f32 = 0.0;

    for (i, &target) in input.iter().enumerate() {
        let out = output[i];

        let desired = if target < 0.0 { 1.0 } else { 0.0 };
        if out == desired {
            // approximately equal
            correct += 1.0;
        }
    }

    correct
}
