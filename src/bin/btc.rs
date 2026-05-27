use anyhow::Result;
use snn_2::{
    btc::{csv::parse_csv, input::encode_btc},
    layer::{IzhikevichLayerBuilder, IzhikevichModel},
    plot::{plot_all_layers, plot_all_layers_raster},
};

fn main() -> Result<()> {
    const T: usize = 30;

    let most_recent = 1779841140;
    // 1779254229 = week ago
    //let data = parse_csv("./data/btcusd_1-min_data.csv", 1779254229)?;
    let data = parse_csv("./data/btcusd_1-min_data.csv", most_recent - 600)?;
    println!("data len: {}", data.len());
    let input = encode_btc(&data);

    //for i in 0..input.len() {
    //    if i == 0 {
    //        continue;
    //    };
    //    let spike = input[i];
    //    let curr_price = &data[i].close;
    //    let prev_price = &data[i - 1].close;
    //    println!(
    //        "Spike: {} | Prev: {} | Curr: {}",
    //        spike, prev_price, curr_price
    //    );
    //}

    let mut model = IzhikevichModel::init();

    let input_dim = input.len();
    println!("Input dim: {}", input_dim);

    let input_layer = IzhikevichLayerBuilder::new()
        .in_n(input_dim)
        .out_n(input_dim)
        .n_conns(9)
        .threshold(10.0)
        .a(0.1)
        .b(1.)
        .c(-45.0)
        .d(12.0)
        .T(T)
        .a_plus(0.01)
        .a_minus(0.012)
        .tau_minus(20.0)
        .tau_plus(20.0)
        .min_weight(-1.0)
        .max_weight(5.0)
        .build();
    let hidden_layer = IzhikevichLayerBuilder::new()
        .in_n(input_dim)
        .out_n(input_dim)
        .n_conns(9)
        .threshold(1.0)
        .a(0.1)
        .b(2.)
        .c(-75.0)
        .d(12.0)
        .T(T)
        .a_plus(0.01)
        .a_minus(0.012)
        .tau_minus(20.0)
        .tau_plus(20.0)
        .min_weight(-5.0)
        .max_weight(5.0)
        .build();
    let output_layer = IzhikevichLayerBuilder::new()
        .in_n(input_dim) // 3x3 grid
        .out_n(input_dim)
        .n_conns(9)
        .threshold(1.0)
        .a(0.1)
        .b(2.)
        .c(-75.0)
        .d(12.0)
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

    let output = model.run(T, input.view());
    println!("output: {}", output);

    plot_all_layers(&model.layers, "charts/btc.png").expect("failed to create plot");
    plot_all_layers_raster(&model.layers, "charts/btc-raster.png")
        .expect("failed to create raster");

    Ok(())
}
