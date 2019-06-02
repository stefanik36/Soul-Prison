package com.stefanik36.soul_prison.data_sets;

import com.stefanik36.soul_prison.builder.ActivationFunctionFactory;
import com.stefanik36.soul_prison.builder.NetworkBuilder;
import com.stefanik36.soul_prison.builder.OutputResolverFactory;
import com.stefanik36.soul_prison.builder.ValidationFunctionFactory;
import com.stefanik36.soul_prison.data.DataResultTuple;
import com.stefanik36.soul_prison.data.TrainData;
import com.stefanik36.soul_prison.model.TestResult;
import com.stefanik36.soul_prison.source.DataSource;
import com.stefanik36.soul_prison.source.RedWineQualityData;
import com.stefanik36.soul_prison.util.ActivationFunction;
import com.stefanik36.soul_prison.util.PlotUtil;
import com.stefanik36.soul_prison.util.TestUtil;
import io.vavr.collection.List;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;

public class RedWineQualityDataTest {

    @Test
    public void getData01() {
        List<DataResultTuple> rwq = DataSource.getRedWineQualityData();
        assertEquals(rwq.size(), 1599);
    }


    @Test
    public void n01() {
        int seed = 666;
        Random random = new Random(seed);
        NetworkBuilder networkBuilder = NetworkBuilder.initFullyConnected(
                ValidationFunctionFactory.classification(),
                11,
                6,
                8, 3, 4
        )
                .setOutputResolver(OutputResolverFactory.onlyTestResults())
                .setActivationFunction(ActivationFunctionFactory.sigmoid())
                .setBias(0.0)
                .setLearningRate(0.3)
                .setMomentum(0.3)
                .setMean(0.0)
                .setRange(1.0);

        List<Double> categoryValues = List.of(3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        TrainData trainData = new TrainData(DataSource.getRedWineQualityData(), random, categoryValues);

        TestResult result = TestUtil.testDifferentRandoms(
                (d, net) -> {
                    net.train(d, 1000);
                    net.test(d);
                    return net.getTestResult();
                },
                networkBuilder,
                trainData,
                random,
                5
        );

        assertEquals(0.5845511482254697, result.getTestAccuracy(), 0.0001);
    }


    @Test
    public void n02() {
        int seed = 666;
        Random random = new Random(seed);
        NetworkBuilder networkBuilder = NetworkBuilder.initFullyConnected(
                ValidationFunctionFactory.classification(),
                11,
                6,
                10, 6, 3, 6
        )
                .setOutputResolver(OutputResolverFactory.onlyTestResults())
                .setActivationFunction(ActivationFunctionFactory.sigmoid())
                .setBias(0.0)
                .setLearningRate(0.3)
                .setMomentum(0.3)
                .setMean(0.0)
                .setRange(1.0);

        List<Double> categoryValues = List.of(3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        TrainData trainData = new TrainData(DataSource.getRedWineQualityData(), random, categoryValues);

        TestResult result = TestUtil.testDifferentRandoms(
                (d, net) -> {
                    net.train(d, 1000);
                    net.test(d);
                    return net.getTestResult();
                },
                networkBuilder,
                trainData,
                random,
                5
        );

        assertEquals(0.5720250521920669, result.getTestAccuracy(), 0.0001);
    }

    @Test
    public void n03() {
        int seed = 666;
        Random random = new Random(seed);
        NetworkBuilder networkBuilder = NetworkBuilder.initFullyConnected(
                ValidationFunctionFactory.classification(),
                11,
                6,
                20, 20
        )
                .setOutputResolver(OutputResolverFactory.onlyTestResults())
                .setActivationFunction(ActivationFunctionFactory.sigmoid())
                .setBias(0.0)
                .setLearningRate(0.3)
                .setMomentum(0.3)
                .setMean(0.0)
                .setRange(1.0);

        List<Double> categoryValues = List.of(3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        TrainData trainData = new TrainData(DataSource.getRedWineQualityData(), random, categoryValues);

        TestResult result = TestUtil.testDifferentRandoms(
                (d, net) -> {
                    net.train(d, 1000);
                    net.test(d);
                    return net.getTestResult();
                },
                networkBuilder,
                trainData,
                random,
                5
        );

        assertEquals(0.5732776617954071, result.getTestAccuracy(), 0.0001);
    }

    @Test
    public void n04() {
        int seed = 666;
        Random random = new Random(seed);
        NetworkBuilder networkBuilder = NetworkBuilder.initFullyConnected(
                ValidationFunctionFactory.classification(),
                11,
                6,
                8, 4, 6
        )
                .setOutputResolver(OutputResolverFactory.onlyTestResults())
                .setActivationFunction(ActivationFunctionFactory.sigmoid())
                .setBias(0.0)
                .setLearningRate(0.3)
                .setMomentum(0.3)
                .setMean(0.0)
                .setRange(1.0);

        List<Double> categoryValues = List.of(3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        TrainData trainData = new TrainData(DataSource.getRedWineQualityData(), random, categoryValues);

        TestResult result = TestUtil.testDifferentRandoms(
                (d, net) -> {
                    net.train(d, 1000);
                    net.test(d);
//                    PlotUtil.plotResult(net.getTrainSummaryList());
                    return net.getTestResult();
                },
                networkBuilder,
                trainData,
                random,
                5
        );
        assertEquals(0.5853862212943632, result.getTestAccuracy(), 0.0001);
    }


}