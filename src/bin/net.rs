use anyhow::Result;
use ndarray::Array2;
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use snn_2::{
    izhikevich::{Izhikevich, IzhikevichBuilder, Logger},
    plot::{plot_izhikievich, plot_raster, plot_weights, plot_weights2},
    plt::Plt,
};

fn main() -> Result<()> {
    //let n_conns = 10;
    //let threshold = 40.0;
    //let a = 0.01; // Time scale of the recovery variable. Smaller -> slower recovery
    //let b = 0.2; // Sensitivty of the recovery variable. Larger -> stronger coupling
    //let c = -65.0; // Reset voltage of membrane
    //let d = 8.0; // Increment of the recovery variable after a spike.

    const T: usize = 200;
    //let mut model = Izhikevich::new(n_neurons, n_conns, threshold, 20.0, 20.0, a, b, c, d, T);

    let n_neurons = 9;
    let mut model = IzhikevichBuilder::new()
        .n_neurons(n_neurons)
        .n_conns(3)
        .threshold(40.0)
        .a(0.02)
        .b(0.2)
        .c(-75.0)
        .d(6.0)
        .a_plus(0.001)
        .a_minus(0.002)
        .tau_minus(20.0)
        .tau_plus(20.0)
        .min_weight(-1.0)
        .max_weight(1.0)
        .T(T)
        .build();

    let steps = (T as f32 / 0.01) as usize;

    let i = vec![0., 0., 0., 0., 4., 0., 0., 0., 0.];

    //let i_t: Vec<f32> = (0..T).map(|t| i).collect();
    let i_t: Vec<Vec<f32>> = vec![i; steps];
    let io = i_t.into_iter().flatten().collect();

    let input: Array2<f32> = Array2::from_shape_vec((steps, n_neurons), io).unwrap();

    //let input = Array2::random((steps, n_neurons), Uniform::new(-6.0, 6.0).unwrap());

    let mut logger = Logger::new(T, 0.1, &model);
    let output = model.run(T as f32, input, &mut logger);

    //Plt::spikes("spikes.png", output)?;

    let rt = plot_izhikievich(&output, "charts/spikes3.png").unwrap();
    plot_raster(&logger.spikes_data, "charts/raster2.png").unwrap();
    plot_weights2(&logger.weights, model.n_conns, "charts/weights.png")
        .expect("Failed to plot weights");
    Ok(())
}
