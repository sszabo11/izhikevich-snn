use anyhow::Result;
use ndarray::{Array1, Array2, array};
use snn_2::{
    layer::{IzhikevichLayerBuilder, IzhikevichModel, Logger},
    plot::{
        plot_all_layers, plot_all_layers_raster, plot_izhikievich, plot_raster, plot_weights,
        plot_weights2,
    },
};

fn main() -> Result<()> {
    const T: usize = 10;

    let mut model = IzhikevichModel::init();

    let input_layer = IzhikevichLayerBuilder::new()
        .in_n(9) // 3x3 grid
        .out_n(9)
        .n_conns(10)
        .threshold(40.0)
        .a(0.01)
        .b(0.2)
        .c(-35.0)
        .d(6.0)
        .T(T)
        .a_plus(0.01)
        .a_minus(0.012)
        .tau_minus(20.0)
        .tau_plus(20.0)
        .min_weight(-5.0)
        .max_weight(5.0)
        .build();
    let hidden_layer = IzhikevichLayerBuilder::new()
        .in_n(9) // 3x3 grid
        .out_n(64)
        .n_conns(200)
        .threshold(40.0)
        .a(0.01)
        .b(0.2)
        .c(-75.0)
        .d(6.0)
        .T(T)
        .a_plus(0.01)
        .a_minus(0.012)
        .tau_minus(20.0)
        .tau_plus(20.0)
        .min_weight(-5.0)
        .max_weight(5.0)
        .build();
    let output_layer = IzhikevichLayerBuilder::new()
        .in_n(64) // 3x3 grid
        .out_n(9)
        .n_conns(400)
        .threshold(40.0)
        .a(0.01)
        .b(0.2)
        .c(-75.0)
        .d(6.0)
        .T(T)
        .a_plus(0.01)
        .a_minus(0.012)
        .tau_minus(20.0)
        .tau_plus(20.0)
        .min_weight(-5.0)
        .max_weight(5.0)
        .build();

    model.add_layer(input_layer);
    model.add_layer(hidden_layer);
    model.add_layer(output_layer);

    let input: Array1<f32> = array![
        1000., 1000., 1000., -100., -100., -100., -100., -100., -100.
    ];

    let output = model.run(T, input);
    println!("output: {}", output);

    plot_all_layers(&model.layers, "charts/layerd.png").expect("fasield to ");
    plot_all_layers_raster(&model.layers, "charts/l_rast.png").expect("fasield to ");
    //let rt = plot_izhikievich(&output, "charts/spikes3.png").unwrap();
    //plot_raster(&logger.spikes_data, "charts/raster2.png").unwrap();
    //plot_weights2(&logger.weights, model.n_conns, "charts/weights.png")
    //    .expect("Failed to plot weights");
    Ok(())
}
