package com.stefanik36.soul_prison.data_sets;

import com.stefanik36.soul_prison.builder.ActivationFunctionFactory;
import com.stefanik36.soul_prison.builder.NetworkBuilder;
import com.stefanik36.soul_prison.builder.OutputResolverFactory;
import com.stefanik36.soul_prison.builder.ValidationFunctionFactory;
import com.stefanik36.soul_prison.data.DataResultTuple;
import com.stefanik36.soul_prison.data.TrainData;
import com.stefanik36.soul_prison.model.TestResult;
import com.stefanik36.soul_prison.source.DataSource;
import com.stefanik36.soul_prison.util.TestUtil;
import io.vavr.collection.List;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;

public class BreastCancerWisconsinDataTest {

    public static void main(String[] args) {
        new BreastCancerWisconsinDataTest().n01();
    }

    @Test
    public void getData01() {
        List<DataResultTuple> rwq = DataSource.getBreastCancerWisconsinData();
        assertEquals(rwq.size(), 699);
    }

    @Test
    public void n01() {
        int seed = 536831334;
        Random random = new Random(seed);

        NetworkBuilder networkBuilder = NetworkBuilder.initFullyConnected(
                ValidationFunctionFactory.classification(),
                9,
                2,
                10, 4
        )
                .setOutputResolver(OutputResolverFactory.onlyTestResults())
                .setActivationFunction(ActivationFunctionFactory.sigmoid())
                .setBias(0.0)
                .setLearningRate(0.3)
                .setMomentum(0.3)
                .setMean(0.0)
                .setRange(1.0);

        List<Double> categoryValues = List.of(2.0, 4.0);
        TrainData trainData = new TrainData(DataSource.getBreastCancerWisconsinData(), random, categoryValues);

        TestResult result = TestUtil.testDifferentRandoms(
                (d, net) -> {
                    net.train(d, 500);
                    net.test(d);
                    return net.getTestResult();
                },
                networkBuilder,
                trainData,
                random,
                seed,
                10
        );

        assertEquals(0.7980861244019138, result.getTestAccuracy(), 0.0001);
    }

}