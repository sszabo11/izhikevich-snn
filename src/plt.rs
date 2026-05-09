use anyhow::Result;
use ndarray::Array2;
use plotters::prelude::*;
use rand::RngExt;

use crate::izhikevich::ModelResponse;

pub struct Plt {}

impl Plt {
    fn new() -> Self {
        Self {}
    }

    pub fn raster(name: &str, spikes: &Array2<u32>) -> Result<()> {
        let path = format!("charts/{}", name);

        //let random_points: Vec<(f64, f64)> = {
        //    let mut rng = rand::rng();

        //    //let mut x_rand = rng.random_range(0.5..0.13);
        //    //let mut y_rand = rng.random_range(0.5..0.13);
        //    let x_iter = (0..5000)
        //        .map(|_| rng.random_range(0.13..0.5))
        //        .collect::<Vec<f64>>();
        //    let y_iter = (0..5000)
        //        .map(|_| rng.random_range(0.13..0.5))
        //        .collect::<Vec<f64>>();

        //    x_iter.into_iter().zip(y_iter).take(5000).collect()
        //};

        //let root = BitMapBackend::new(&path, (1040, 980)).into_drawing_area();
        //root.fill(&WHITE)?;
        //let mut chart = ChartBuilder::on(&root)
        //    .caption("Raster", ("sans-serif", 50).into_font())
        //    .margin(5)
        //    .x_label_area_size(30)
        //    .y_label_area_size(30)
        //    .build_cartesian_2d(-1f32..1f32, -0.1f32..1f32)?;

        //chart.configure_mesh().draw()?;

        //let areas = root.split_by_breakpoints([944], [80]);

        //let mut scatter_ctx = ChartBuilder::on(&areas[2])
        //    .x_label_area_size(40)
        //    .y_label_area_size(40)
        //    .build_cartesian_2d(0f64..1f64, 0f64..1f64)?;
        //scatter_ctx
        //    .configure_mesh()
        //    .disable_x_mesh()
        //    .disable_y_mesh()
        //    .draw()?;
        //scatter_ctx.draw_series(
        //    random_points
        //        .iter()
        //        .map(|(x, y)| Circle::new((*x, *y), 2, GREEN.filled())),
        //)?;

        //chart
        //    .draw_series(LineSeries::new(
        //        (-50..=50).map(|x| x as f32 / 50.0).map(|x| (x, x * x)),
        //        &RED,
        //    ))?
        //    .label("y = x^2")
        //    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        //chart
        //    .configure_series_labels()
        //    .background_style(&WHITE.mix(0.8))
        //    .border_style(&BLACK)
        //    .draw()?;

        //root.present()?;
        let (n_timesteps, n_neurons) = spikes.dim();

        let max_time = n_timesteps as f64;
        let root = SVGBackend::new(&path, (1200, 800)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("SNN Raster Plot", ("sans-serif", 50))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(0f64..max_time, 0..n_neurons)?;

        chart.configure_mesh().draw()?;

        let spike_points: Vec<(f64, usize)> = spikes
            .indexed_iter() // gives (index, &value)
            .filter(|&((t, n), &val)| val > 0)
            .map(|((t, n), _)| (t as f64, n))
            .collect();
        // Group spikes per neuron
        chart.draw_series(PointSeries::of_element(
            spike_points,
            2,
            ShapeStyle::from(&RED).filled(),
            &|c, s, _| Circle::new(c, s, RED.filled()),
        ))?;

        root.present()?;
        Ok(())
    }

    pub fn spikes(name: &str, res: ModelResponse) -> Result<()> {
        let path = format!("charts/{}", name);
        let (n_timesteps, n_neurons) = res.v_values.dim();

        let max_time = n_timesteps as f64;
        let root = BitMapBackend::new(&path, (1200, 800)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Spikes", ("sans-serif", 50))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(0f64..max_time, 0..n_neurons)?;

        chart.configure_mesh().draw()?;

        let spike_points: Vec<(f64, usize)> = res
            .v_values
            .indexed_iter() // gives (index, &value)
            .map(|((t, n), _)| (t as f64, n))
            .collect();
        chart
            .draw_series(LineSeries::new(spike_points, &RED))?
            .label("y = x^2")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        root.present()?;
        Ok(())
    }
}
