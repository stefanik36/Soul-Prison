package com.stefanik36.soul_prison.data_sets;

import com.stefanik36.soul_prison.builder.ActivationFunctionFactory;
import com.stefanik36.soul_prison.builder.NetworkBuilder;
import com.stefanik36.soul_prison.builder.OutputResolverFactory;
import com.stefanik36.soul_prison.builder.ValidationFunctionFactory;
import com.stefanik36.soul_prison.data.TrainData;
import com.stefanik36.soul_prison.model.TestResult;
import com.stefanik36.soul_prison.source.DataSource;
import com.stefanik36.soul_prison.util.TestUtil;
import io.vavr.collection.List;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;

public class WineDataTest {

    @Test
    public void wineDataSigmoidMomentum01() {
        int seed = 666;
        Random random = new Random(seed);

        NetworkBuilder networkBuilder = NetworkBuilder
                .initFullyConnected(ValidationFunctionFactory.classification(), 13, 3, 12)
                .setBias(0.0)
                .setLearningRate(0.8)
                .setMomentum(0.3)
                .setActivationFunction(ActivationFunctionFactory.sigmoid())
                .setOutputResolver(OutputResolverFactory.onlyTestResults())
                .setMean(0.0)
                .setRange(1.0);

        TrainData data = new TrainData(DataSource.getWineData(), random, List.of(1.0, 2.0, 3.0));
        TestResult result = TestUtil.testDifferentRandoms(
                (td, net) -> {
                    net.train(td, 300);
                    net.test(td);
                    return net.getTestResult();
                },
                networkBuilder,
                data,
                random,
                20
        );
        assertEquals(0.6764150943396225, result.getTestAccuracy(), 0.001);

//        List<TrainSummary> trainR = network.getTrainSummaryList();
//        PlotUtil.plotResult(trainR, tr);
    }


    @Test
    public void WineDataSigmoid() {
        int seed = 66;
        Random random = new Random(seed);
        NetworkBuilder networkBuilder = NetworkBuilder
                .initFullyConnected(ValidationFunctionFactory.classification(), 13, 3, 12)
                .setBias(0.0)
                .setLearningRate(0.3)
                .setMomentum(0.0)
                .setActivationFunction(ActivationFunctionFactory.sigmoid())
                .setOutputResolver(OutputResolverFactory.onlyTestResults())
                .setMean(0.0)
                .setRange(1.0);

        TrainData data = new TrainData(DataSource.getWineData(), random, List.of(1.0, 2.0, 3.0));
        TestResult result = TestUtil.testDifferentRandoms(
                (td, net) -> {
                    net.train(td, 400);
                    net.test(td);
                    return net.getTestResult();
                },
                networkBuilder,
                data,
                random,
                20
        );

        assertEquals(0.6698113207547169, result.getTestAccuracy(), 0.001);
    }


    @Test
    public void wineDataSigmoidWithBias() {
        int seed = 66;
        Random random = new Random(seed);
        NetworkBuilder networkBuilder = NetworkBuilder
                .initFullyConnected(ValidationFunctionFactory.classification(), 13, 3, 12)
                .setBias(1.0)
                .setLearningRate(0.3)
                .setMomentum(0.0)
                .setActivationFunction(ActivationFunctionFactory.sigmoid())
                .setOutputResolver(OutputResolverFactory.onlyTestResults())
                .setMean(0.0)
                .setRange(1.0);

        TrainData data = new TrainData(DataSource.getWineData(), random, List.of(1.0, 2.0, 3.0));
        TestResult result = TestUtil.testDifferentRandoms(
                (td, net) -> {
                    net.train(td, 300);
                    net.test(td);
                    return net.getTestResult();
                },
                networkBuilder,
                data,
                random,
                20
        );

        assertEquals(0.6650943396226416, result.getTestAccuracy(), 0.001);
    }


    @Test
    public void wineDataSigmoidWithBiasAndMomentum() {
        int seed = 66;
        Random random = new Random(seed);
        NetworkBuilder networkBuilder = NetworkBuilder
                .initFullyConnected(ValidationFunctionFactory.classification(), 13, 3, 12)
                .setBias(1.0)
                .setLearningRate(0.3)
                .setMomentum(0.8)
                .setActivationFunction(ActivationFunctionFactory.sigmoid())
                .setOutputResolver(OutputResolverFactory.onlyTestResults())
                .setMean(0.0)
                .setRange(1.0);

        TrainData data = new TrainData(DataSource.getWineData(), random, List.of(1.0, 2.0, 3.0));
        TestResult result = TestUtil.testDifferentRandoms(
                (td, net) -> {
                    net.train(td, 500);
                    net.test(td);
                    return net.getTestResult();
                },
                networkBuilder,
                data,
                random,
                20
        );

        assertEquals(0.6283018867924529, result.getTestAccuracy(), 0.001);
    }

    @Test
    public void wineDataSigmoidMomentum() {
        int seed = 666;
        Random random = new Random(seed);
        NetworkBuilder networkBuilder = NetworkBuilder
                .initFullyConnected(ValidationFunctionFactory.classification(), 13, 3, 13, 8)
                .setBias(0.0)
                .setLearningRate(0.3)
                .setMomentum(0.3)
                .setActivationFunction(ActivationFunctionFactory.sigmoid())
                .setOutputResolver(OutputResolverFactory.onlyTestResults())
                .setMean(0.0)
                .setRange(0.5);

        TrainData data = new TrainData(DataSource.getWineData(), random, List.of(1.0, 2.0, 3.0));

        TestResult result = TestUtil.testDifferentRandoms(
                (td, net) -> {
                    net.train(td, 2000);
                    net.test(td);
                    return net.getTestResult();
                },
                networkBuilder,
                data,
                random,
                20
        );

        assertEquals(0.7264150943396228, result.getTestAccuracy(), 0.001);
    }


}
