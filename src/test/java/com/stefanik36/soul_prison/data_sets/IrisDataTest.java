package com.stefanik36.soul_prison.data_sets;

import com.stefanik36.soul_prison.builder.ActivationFunctionFactory;
import com.stefanik36.soul_prison.builder.NetworkBuilder;
import com.stefanik36.soul_prison.builder.OutputResolverFactory;
import com.stefanik36.soul_prison.builder.ValidationFunctionFactory;
import com.stefanik36.soul_prison.data.TrainData;
import com.stefanik36.soul_prison.model.TestResult;
import com.stefanik36.soul_prison.network_components.layer.Step;
import com.stefanik36.soul_prison.network_components.neuron.NeuronNode;
import com.stefanik36.soul_prison.source.DataSource;
import com.stefanik36.soul_prison.util.TestUtil;
import io.vavr.collection.List;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;

public class IrisDataTest {


    @Test
    public void irisDataSigmoid() {
        Random random = new Random(66);
        NetworkBuilder networkBuilder = NetworkBuilder.initFullyConnected(ValidationFunctionFactory.classification(), 4, 3, 4, 3)
                .setBias(0.0)
                .setLearningRate(0.3)
                .setOutputResolver(OutputResolverFactory.onlyTestResults())
                .setMomentum(0.0);


        List<Double> categoryValues = List.of(1.0, 2.0, 3.0);
        TrainData trainData = new TrainData(DataSource.getIrisData(), random, categoryValues).setTest(0.3);

        TestResult result = TestUtil.testDifferentRandoms((td, net) -> {
                    net.train(td, 200);
                    net.test(td);
                    return net.getTestResult();
                },
                networkBuilder,
                trainData,
                random,
                20
        );


        assertEquals(0.9533333333333334, result.getTestAccuracy(), 0.001);
    }


    @Test
    public void irisDataSigmoidWithBias() {
        Random random = new Random(66);
        NetworkBuilder networkBuilder = NetworkBuilder.initFullyConnected(ValidationFunctionFactory.classification(), 4, 3, 4, 3)
                .setBias(1.0)
                .setLearningRate(0.3)
                .setOutputResolver(OutputResolverFactory.onlyTestResults())
                .setMomentum(0.0);

        List<Double> categoryValues = List.of(1.0, 2.0, 3.0);
        TrainData trainData = new TrainData(DataSource.getIrisData(), random, categoryValues).setTest(0.3);

        TestResult result = TestUtil.testDifferentRandoms((td, net) -> {
                    net.train(td, 100);
                    net.test(td);
                    return net.getTestResult();
                },
                networkBuilder,
                trainData,
                random,
                20
        );


        assertEquals(0.928888888888889, result.getTestAccuracy(), 0.001);
    }


    @Test
    public void irisDataSigmoidWithBiasAndMomentum() {
        Random random = new Random(66);
        NetworkBuilder networkBuilder = NetworkBuilder.initFullyConnected(ValidationFunctionFactory.classification(), 4, 3, 4, 3)
                .setBias(1.0)
                .setLearningRate(0.3)
                .setOutputResolver(OutputResolverFactory.onlyTestResults())
                .setMomentum(0.3);


        List<Double> categoryValues = List.of(1.0, 2.0, 3.0);
        TrainData trainData = new TrainData(DataSource.getIrisData(), random, categoryValues).setTest(0.3);


        TestResult result = TestUtil.testDifferentRandoms((td, net) -> {
                    net.train(td, 100);
                    net.test(td);
                    return net.getTestResult();
                },
                networkBuilder,
                trainData,
                random,
                20
        );


        assertEquals(0.9244444444444445, result.getTestAccuracy(), 0.001);

    }

    @Test
    public void irisDataSigmoidFullyConnectedByBuilder() {
        Random random = new Random(66);
        NetworkBuilder networkBuilder = NetworkBuilder.initFullyConnected(ValidationFunctionFactory.classification(), 4, 3, 4, 3)
                .setBias(0.0)
                .setLearningRate(0.3)
                .setMomentum(0.0)
                .setActivationFunction(ActivationFunctionFactory.sigmoid())
                .setOutputResolver(OutputResolverFactory.onlyTestResults())
                .setMean(0.0)
                .setRange(1.0);

        List<Double> categoryValues = List.of(1.0, 2.0, 3.0);
        TrainData trainData = new TrainData(DataSource.getIrisData(), random, categoryValues).setTest(0.3);


        TestResult result = TestUtil.testDifferentRandoms((td, net) -> {
                    net.train(td, 200);
                    net.test(td);
                    return net.getTestResult();
                },
                networkBuilder,
                trainData,
                random,
                20
        );


        assertEquals(0.9533333333333334, result.getTestAccuracy(), 0.001);
    }


    @Test
    public void irisDataSigmoidFullyConnectedByBuilderDeleteConnection() {

        Random random = new Random(66);
        NetworkBuilder networkBuilder = NetworkBuilder.initFullyConnected(ValidationFunctionFactory.classification(), 4, 3, 4, 3)
                .setBias(0.0)
                .setLearningRate(0.3)
                .setMomentum(0.0)
                .setActivationFunction(ActivationFunctionFactory.sigmoid())
                .setOutputResolver(OutputResolverFactory.onlyTestResults())
                .setMean(0.0)
                .setRange(1.0);


        List<Double> categoryValues = List.of(1.0, 2.0, 3.0);
        TrainData trainData = new TrainData(DataSource.getIrisData(), random, categoryValues).setTest(0.3);
        TestResult result = TestUtil.testDifferentRandoms((td, net) -> {
                    Step output = net.getSteps().last();
                    output.getNeuronsWithBias()
                            .filter(n -> n instanceof NeuronNode)
                            .map(n -> (NeuronNode) n).subSequence(0,1)
                            .forEach(
                                    n -> net.getAllNeurons()
                                            .forEach(n::removeConnection)
                            );
                    net.train(td, 100);
                    net.test(td);
                    return net.getTestResult();
                },
                networkBuilder,
                trainData,
                random,
                20
        );

        assertEquals(0.6211111111111111, result.getTestAccuracy(), 0.001);

    }


//    @Test
//    public void irisDataSigmoidNotFullyConnected() {
//        Random random = new Random(66);
//        Supplier<Double> weightSupplier = () -> (random.nextDouble() - 0.5) * 2.0;
//
//        /*
//         * input layer
//         */
//        NeuronBias linB = new NeuronBias(0.0);
//        NeuronInput lin00 = new NeuronInput();
//        NeuronInput lin01 = new NeuronInput();
//        NeuronInput lin02 = new NeuronInput();
//        NeuronInput lin03 = new NeuronInput();
//
//        /*
//         * layer 00
//         */
//        NeuronBias l0nB = new NeuronBias(0.0);
//        NeuronNodeWithConnections l0n00 = new NeuronNodeWithConnections(
//                "l0_n00",
//                List.of(
//                        new NeuronConnection(linB, weightSupplier.get()),
//                        new NeuronConnection(lin00, weightSupplier.get()),
//                        new NeuronConnection(lin02, weightSupplier.get())
//                ),
//                Option.of(ActivationFunctionFactoryNew.sigmoid()),
//                0.3,
//                0.4
//        );
//
//        NeuronNodeWithConnections l0n01 = new NeuronNodeWithConnections(
//                "l0_n01",
//                List.of(
//                        new NeuronConnection(linB, weightSupplier.get()),
//                        new NeuronConnection(lin00, weightSupplier.get()),
//                        new NeuronConnection(lin01, weightSupplier.get()),
//                        new NeuronConnection(lin03, weightSupplier.get())
//                ),
//                Option.of(ActivationFunctionFactoryNew.sigmoid()),
//                0.3,
//                0.4
//        );
//
//        NeuronNodeWithConnections l0n02 = new NeuronNodeWithConnections(
//                "l0_n02",
//                List.of(
//                        new NeuronConnection(lin02, weightSupplier.get()),
//                        new NeuronConnection(lin03, weightSupplier.get())
//                ),
//                Option.of(ActivationFunctionFactoryNew.sigmoid()),
//                0.3,
//                0.4
//        );
//        NeuronNodeWithConnections l0n03 = new NeuronNodeWithConnections(
//                "l0_n03",
//                List.of(
//                        new NeuronConnection(lin01, weightSupplier.get()),
//                        new NeuronConnection(lin03, weightSupplier.get())
//                ),
//                Option.of(ActivationFunctionFactoryNew.sigmoid()),
//                0.3,
//                0.4
//        );
//
//        /*
//         * layer 01
//         */
//
//        NeuronBias l1nB = new NeuronBias(0.0);
//        NeuronNodeWithConnections l1n04 = new NeuronNodeWithConnections(
//                "l1_n04",
//                List.of(
//                        new NeuronConnection(l0nB, weightSupplier.get()),
//                        new NeuronConnection(l0n00, weightSupplier.get()),
//                        new NeuronConnection(l0n01, weightSupplier.get())
//                ),
//                Option.of(ActivationFunctionFactoryNew.sigmoid()),
//                0.3,
//                0.4
//        );
//        NeuronNodeWithConnections l1n05 = new NeuronNodeWithConnections(
//                "l1_n05",
//                List.of(
//                        new NeuronConnection(l0n00, weightSupplier.get()),
//                        new NeuronConnection(l0n02, weightSupplier.get())
//                ),
//                Option.of(ActivationFunctionFactoryNew.sigmoid()),
//                0.3,
//                0.4
//        );
//        NeuronNodeWithConnections l1n06 = new NeuronNodeWithConnections(
//                "l1_n06",
//                List.of(
//                        new NeuronConnection(l0nB, weightSupplier.get()),
//                        new NeuronConnection(l0n00, weightSupplier.get()),
//                        new NeuronConnection(l0n01, weightSupplier.get()),
//                        new NeuronConnection(l0n02, weightSupplier.get()),
//                        new NeuronConnection(l0n03, weightSupplier.get())
//                ),
//                Option.of(ActivationFunctionFactoryNew.sigmoid()),
//                0.3,
//                0.4
//        );
//
//        /*
//         * layer 01 (output)
//         */
//        NeuronBias l2nB = new NeuronBias(0.0);
//        NeuronNodeWithConnections l2n07 = new NeuronNodeWithConnections(
//                "l1_n07",
//                List.of(
//                        new NeuronConnection(l1nB, weightSupplier.get()),
//                        new NeuronConnection(l1n04, weightSupplier.get()),
//                        new NeuronConnection(l1n05, weightSupplier.get()),
//                        new NeuronConnection(l1n06, weightSupplier.get())
//                ),
//                Option.none(),
//                0.3,
//                0.4
//        );
//        NeuronNodeWithConnections l2n08 = new NeuronNodeWithConnections(
//                "l1_n08",
//                List.of(
//                        new NeuronConnection(l1nB, weightSupplier.get()),
//                        new NeuronConnection(l1n04, weightSupplier.get()),
//                        new NeuronConnection(l1n05, weightSupplier.get()),
//                        new NeuronConnection(l1n06, weightSupplier.get())
//                ),
//                Option.none(),
//                0.3,
//                0.4
//        );
//        NeuronNodeWithConnections l2n09 = new NeuronNodeWithConnections(
//                "l1_n09",
//                List.of(
//                        new NeuronConnection(l1nB, weightSupplier.get()),
//                        new NeuronConnection(l1n04, weightSupplier.get()),
//                        new NeuronConnection(l1n05, weightSupplier.get())
//                ),
//                Option.none(),
//                0.3,
//                0.4
//        );
//
//        NetworkWithConnections network = new NetworkWithConnections(
//                new StepInput(List.of(lin00, lin01, lin02, lin03), linB),
//                List.of(
//                        new StepWithConnections(
//                                List.of(l0n00, l0n01, l0n02, l0n03),
//                                l0nB
//                        ),
//                        new StepWithConnections(
//                                List.of(l1n04, l1n05, l1n06),
//                                l1nB
//                        )
//                ),
//                new StepWithConnections(
//                        List.of(l2n07, l2n08, l2n09),
//                        l2nB
//                ),
//                ValidationFunctionFactory.classification()
//        );
//
//        List<Double> categoryValues = List.of(1.0, 2.0, 3.0);
//        List<DataResultTuple> iData = ListUtil.shuffle(DataSource.getIrisData(), random);
//
//        int div = ((Double) (iData.size() * 0.3)).intValue();
//        List<DataResultTuple> trainSet = iData.subSequence(div);
//
//
//        List.range(0, 100).map(v -> ListUtil.shuffle(trainSet, random)).forEach(ts -> ts.forEach(wd -> {
//            Input input = new Input(wd.getData()).getNormalized();
//            network.propagate(input);
//
//            network.backPropagate(new Input(wd.getClassificationResult(categoryValues)));
//            network.updateWeights();
//
//        }));
//
//
//        List<DataResultTuple> testSet = iData.subSequence(0, div);
//        AtomicInteger valid = new AtomicInteger();
//        testSet.forEach(wd -> {
//
//            network.propagate(new Input(wd.getData()).getNormalized());
//            Input result = network.getResult();
//
//
//            String correct = "_";
//            if (result.getValues().indexOf(result.getValues().max().get()) == wd.getClassificationResult(categoryValues).indexOf(wd.getClassificationResult(categoryValues).max().get())) {
//                valid.getAndIncrement();
//                correct = "OK";
//            }
//            System.out.println(
//                    "shouldBe: " + wd.getClassificationResult(categoryValues)
//                            + " is: " + result.getValues().map(v -> FormatUtil.DECIMAL_FORMAT.format(v))
//                            + " " + correct
//            );
//
//        });
//
//        double result = ((double) valid.get() / (double) testSet.size());
//        System.out.println("r: " + result);
//        assertEquals(0.5777777777777777, result, 0.001);
//    }
//
//
//    @Test
//    public void irisDataSigmoidNotFullyConnected02() {
//        Random random = new Random(66);
//        Supplier<Double> weightSupplier = () -> (random.nextDouble() - 0.5) * 2.0;
//
//        /*
//         * input layer
//         */
//        NeuronBias linB = new NeuronBias(0.0);
//        NeuronInput lin00 = new NeuronInput();
//        NeuronInput lin01 = new NeuronInput();
//        NeuronInput lin02 = new NeuronInput();
//        NeuronInput lin03 = new NeuronInput();
//
//        /*
//         * layer 00
//         */
//        NeuronBias l0nB = new NeuronBias(0.0);
//        NeuronNodeWithConnections l0n00 = new NeuronNodeWithConnections(
//                "l0_n00",
//                List.of(
//                        new NeuronConnection(linB, weightSupplier.get()),
//                        new NeuronConnection(lin00, weightSupplier.get()),
//                        new NeuronConnection(lin02, weightSupplier.get())
//                ),
//                Option.of(ActivationFunctionFactoryNew.sigmoid()),
//                0.3,
//                0.4
//        );
//
//        NeuronNodeWithConnections l0n01 = new NeuronNodeWithConnections(
//                "l0_n01",
//                List.of(
//                        new NeuronConnection(linB, weightSupplier.get()),
//                        new NeuronConnection(lin00, weightSupplier.get()),
//                        new NeuronConnection(lin01, weightSupplier.get()),
//                        new NeuronConnection(lin03, weightSupplier.get())
//                ),
//                Option.of(ActivationFunctionFactoryNew.sigmoid()),
//                0.3,
//                0.4
//        );
//
//        NeuronNodeWithConnections l0n02 = new NeuronNodeWithConnections(
//                "l0_n02",
//                List.of(
//                        new NeuronConnection(lin02, weightSupplier.get()),
//                        new NeuronConnection(lin03, weightSupplier.get())
//                ),
//                Option.of(ActivationFunctionFactoryNew.sigmoid()),
//                0.3,
//                0.4
//        );
//        NeuronNodeWithConnections l0n03 = new NeuronNodeWithConnections(
//                "l0_n03",
//                List.of(
//                        new NeuronConnection(lin01, weightSupplier.get()),
//                        new NeuronConnection(lin03, weightSupplier.get())
//                ),
//                Option.of(ActivationFunctionFactoryNew.sigmoid()),
//                0.3,
//                0.4
//        );
//
//        /*
//         * layer 01
//         */
//
//        NeuronBias l1nB = new NeuronBias(0.0);
//        NeuronNodeWithConnections l1n04 = new NeuronNodeWithConnections(
//                "l1_n04",
//                List.of(
//                        new NeuronConnection(l0nB, weightSupplier.get()),
//                        new NeuronConnection(l0n00, weightSupplier.get()),
//                        new NeuronConnection(l0n01, weightSupplier.get())
//                ),
//                Option.of(ActivationFunctionFactoryNew.sigmoid()),
//                0.3,
//                0.4
//        );
//        NeuronNodeWithConnections l1n05 = new NeuronNodeWithConnections(
//                "l1_n05",
//                List.of(
//                        new NeuronConnection(l0n00, weightSupplier.get()),
//                        new NeuronConnection(l0n02, weightSupplier.get())
//                ),
//                Option.of(ActivationFunctionFactoryNew.sigmoid()),
//                0.3,
//                0.4
//        );
//        NeuronNodeWithConnections l1n06 = new NeuronNodeWithConnections(
//                "l1_n06",
//                List.of(
//                        new NeuronConnection(l0nB, weightSupplier.get()),
//                        new NeuronConnection(l0n00, weightSupplier.get()),
//                        new NeuronConnection(l0n01, weightSupplier.get()),
//                        new NeuronConnection(l0n02, weightSupplier.get()),
//                        new NeuronConnection(l0n03, weightSupplier.get())
//                ),
//                Option.of(ActivationFunctionFactoryNew.sigmoid()),
//                0.3,
//                0.4
//        );
//
//        /*
//         * layer 01 (output)
//         */
//        NeuronBias l2nB = new NeuronBias(0.0);
//        NeuronNodeWithConnections l2n07 = new NeuronNodeWithConnections(
//                "l1_n07",
//                List.of(
//                        new NeuronConnection(l1nB, weightSupplier.get()),
//                        new NeuronConnection(l1n04, weightSupplier.get()),
//                        new NeuronConnection(l1n05, weightSupplier.get()),
//                        new NeuronConnection(l1n06, weightSupplier.get())
//                ),
//                Option.none(),
//                0.3,
//                0.4
//        );
//        NeuronNodeWithConnections l2n08 = new NeuronNodeWithConnections(
//                "l1_n08",
//                List.of(
//                        new NeuronConnection(l1nB, weightSupplier.get()),
//                        new NeuronConnection(l1n04, weightSupplier.get()),
//                        new NeuronConnection(l1n05, weightSupplier.get()),
//                        new NeuronConnection(l1n06, weightSupplier.get())
//                ),
//                Option.none(),
//                0.3,
//                0.4
//        );
//        NeuronNodeWithConnections l2n09 = new NeuronNodeWithConnections(
//                "l1_n09",
//                List.of(
//                        new NeuronConnection(l1nB, weightSupplier.get()),
//                        new NeuronConnection(l1n04, weightSupplier.get()),
//                        new NeuronConnection(l1n05, weightSupplier.get()),
//                        new NeuronConnection(l1n06, weightSupplier.get())
//                ),
//                Option.none(),
//                0.3,
//                0.4
//        );
//
//        NetworkWithConnections network = new NetworkWithConnections(
//                new StepInput(List.of(lin00, lin01, lin02, lin03), linB),
//                List.of(
//                        new StepWithConnections(
//                                List.of(l0n00, l0n01, l0n02, l0n03),
//                                l0nB
//                        ),
//                        new StepWithConnections(
//                                List.of(l1n04, l1n05, l1n06),
//                                l1nB
//                        )
//                ),
//                new StepWithConnections(
//                        List.of(l2n07, l2n08, l2n09),
//                        l2nB
//                ),
//                ValidationFunctionFactory.classification()
//        );
//
//        List<Double> categoryValues = List.of(1.0, 2.0, 3.0);
//        List<DataResultTuple> iData = ListUtil.shuffle(DataSource.getIrisData(), random);
//
//        int div = ((Double) (iData.size() * 0.3)).intValue();
//        List<DataResultTuple> trainSet = iData.subSequence(div);
//
//
//        List.range(0, 100).map(v -> ListUtil.shuffle(trainSet, random)).forEach(ts -> ts.forEach(wd -> {
//            Input input = new Input(wd.getData()).getNormalized();
//            network.propagate(input);
//
//            network.backPropagate(new Input(wd.getClassificationResult(categoryValues)));
//            network.updateWeights();
//
//        }));
//
//
//        List<DataResultTuple> testSet = iData.subSequence(0, div);
//        AtomicInteger valid = new AtomicInteger();
//        testSet.forEach(wd -> {
//
//            network.propagate(new Input(wd.getData()).getNormalized());
//            Input result = network.getResult();
//
//
//            String correct = "_";
//            if (result.getValues().indexOf(result.getValues().max().get()) == wd.getClassificationResult(categoryValues).indexOf(wd.getClassificationResult(categoryValues).max().get())) {
//                valid.getAndIncrement();
//                correct = "OK";
//            }
//            System.out.println(
//                    "shouldBe: " + wd.getClassificationResult(categoryValues)
//                            + " is: " + result.getValues().map(v -> FormatUtil.DECIMAL_FORMAT.format(v))
//                            + " " + correct
//            );
//
//        });
//
//        double result = ((double) valid.get() / (double) testSet.size());
//        System.out.println("r: " + result);
//        assertEquals(0.9555555555555556, result, 0.001);
//    }
//
//    @Test
//    public void irisDataSigmoidFullyConnected() {
//        Random random = new Random(66);
//        Supplier<Double> weightSupplier = () -> (random.nextDouble() - 0.5) * 2.0;
//
//        /*
//         * input layer
//         */
//        NeuronBias linB = new NeuronBias(0.0);
//        NeuronInput lin00 = new NeuronInput();
//        NeuronInput lin01 = new NeuronInput();
//        NeuronInput lin02 = new NeuronInput();
//        NeuronInput lin03 = new NeuronInput();
//
//        /*
//         * layer 00
//         */
//        NeuronBias l0nB = new NeuronBias(0.0);
//        NeuronNodeWithConnections l0n00 = new NeuronNodeWithConnections(
//                "l0_n00",
//                List.of(
//                        new NeuronConnection(linB, weightSupplier.get()),
//                        new NeuronConnection(lin00, weightSupplier.get()),
//                        new NeuronConnection(lin01, weightSupplier.get()),
//                        new NeuronConnection(lin02, weightSupplier.get()),
//                        new NeuronConnection(lin03, weightSupplier.get())
//                ),
//                Option.of(ActivationFunctionFactoryNew.sigmoid()),
//                0.3,
//                0.0
//        );
//
//        NeuronNodeWithConnections l0n01 = new NeuronNodeWithConnections(
//                "l0_n01",
//                List.of(
//                        new NeuronConnection(linB, weightSupplier.get()),
//                        new NeuronConnection(lin00, weightSupplier.get()),
//                        new NeuronConnection(lin01, weightSupplier.get()),
//                        new NeuronConnection(lin02, weightSupplier.get()),
//                        new NeuronConnection(lin03, weightSupplier.get())
//                ),
//                Option.of(ActivationFunctionFactoryNew.sigmoid()),
//                0.3,
//                0.0
//        );
//
//        NeuronNodeWithConnections l0n02 = new NeuronNodeWithConnections(
//                "l0_n02",
//                List.of(
//                        new NeuronConnection(linB, weightSupplier.get()),
//                        new NeuronConnection(lin00, weightSupplier.get()),
//                        new NeuronConnection(lin01, weightSupplier.get()),
//                        new NeuronConnection(lin02, weightSupplier.get()),
//                        new NeuronConnection(lin03, weightSupplier.get())
//                ),
//                Option.of(ActivationFunctionFactoryNew.sigmoid()),
//                0.3,
//                0.0
//        );
//        NeuronNodeWithConnections l0n03 = new NeuronNodeWithConnections(
//                "l0_n03",
//                List.of(
//                        new NeuronConnection(linB, weightSupplier.get()),
//                        new NeuronConnection(lin00, weightSupplier.get()),
//                        new NeuronConnection(lin01, weightSupplier.get()),
//                        new NeuronConnection(lin02, weightSupplier.get()),
//                        new NeuronConnection(lin03, weightSupplier.get())
//                ),
//                Option.of(ActivationFunctionFactoryNew.sigmoid()),
//                0.3,
//                0.0
//        );
//
//        /*
//         * layer 01
//         */
//
//        NeuronBias l1nB = new NeuronBias(0.0);
//        NeuronNodeWithConnections l1n04 = new NeuronNodeWithConnections(
//                "l1_n04",
//                List.of(
//                        new NeuronConnection(l0nB, weightSupplier.get()),
//                        new NeuronConnection(l0n00, weightSupplier.get()),
//                        new NeuronConnection(l0n01, weightSupplier.get()),
//                        new NeuronConnection(l0n02, weightSupplier.get()),
//                        new NeuronConnection(l0n03, weightSupplier.get())
//                ),
//                Option.of(ActivationFunctionFactoryNew.sigmoid()),
//                0.3,
//                0.0
//        );
//        NeuronNodeWithConnections l1n05 = new NeuronNodeWithConnections(
//                "l1_n05",
//                List.of(
//                        new NeuronConnection(l0nB, weightSupplier.get()),
//                        new NeuronConnection(l0n00, weightSupplier.get()),
//                        new NeuronConnection(l0n01, weightSupplier.get()),
//                        new NeuronConnection(l0n02, weightSupplier.get()),
//                        new NeuronConnection(l0n03, weightSupplier.get())
//                ),
//                Option.of(ActivationFunctionFactoryNew.sigmoid()),
//                0.3,
//                0.0
//        );
//        NeuronNodeWithConnections l1n06 = new NeuronNodeWithConnections(
//                "l1_n06",
//                List.of(
//                        new NeuronConnection(l0nB, weightSupplier.get()),
//                        new NeuronConnection(l0n00, weightSupplier.get()),
//                        new NeuronConnection(l0n01, weightSupplier.get()),
//                        new NeuronConnection(l0n02, weightSupplier.get()),
//                        new NeuronConnection(l0n03, weightSupplier.get())
//                ),
//                Option.of(ActivationFunctionFactoryNew.sigmoid()),
//                0.3,
//                0.0
//        );
//
//        /*
//         * layer 01 (output)
//         */
//        NeuronBias l2nB = new NeuronBias(0.0);
//        NeuronNodeWithConnections l2n07 = new NeuronNodeWithConnections(
//                "l1_n07",
//                List.of(
//                        new NeuronConnection(l1nB, weightSupplier.get()),
//                        new NeuronConnection(l1n04, weightSupplier.get()),
//                        new NeuronConnection(l1n05, weightSupplier.get()),
//                        new NeuronConnection(l1n06, weightSupplier.get())
//                ),
//                Option.of(ActivationFunctionFactoryNew.sigmoid()),
//                0.3,
//                0.0
//        );
//        NeuronNodeWithConnections l2n08 = new NeuronNodeWithConnections(
//                "l1_n08",
//                List.of(
//                        new NeuronConnection(l1nB, weightSupplier.get()),
//                        new NeuronConnection(l1n04, weightSupplier.get()),
//                        new NeuronConnection(l1n05, weightSupplier.get()),
//                        new NeuronConnection(l1n06, weightSupplier.get())
//                ),
//                Option.none(),
//                0.3,
//                0.0
//        );
//        NeuronNodeWithConnections l2n09 = new NeuronNodeWithConnections(
//                "l1_n09",
//                List.of(
//                        new NeuronConnection(l1nB, weightSupplier.get()),
//                        new NeuronConnection(l1n04, weightSupplier.get()),
//                        new NeuronConnection(l1n05, weightSupplier.get()),
//                        new NeuronConnection(l1n06, weightSupplier.get())
//                ),
//                Option.none(),
//                0.3,
//                0.0
//        );
//
//        NetworkWithConnections network = new NetworkWithConnections(
//                new StepInput(List.of(lin00, lin01, lin02, lin03), linB),
//                List.of(
//                        new StepWithConnections(
//                                List.of(l0n00, l0n01, l0n02, l0n03),
//                                l0nB
//                        ),
//                        new StepWithConnections(
//                                List.of(l1n04, l1n05, l1n06),
//                                l1nB
//                        )
//                ),
//                new StepWithConnections(
//                        List.of(l2n07, l2n08, l2n09),
//                        l2nB
//                ),
//                ValidationFunctionFactory.classification()
//        );
//
//        List<Double> categoryValues = List.of(1.0, 2.0, 3.0);
//        List<DataResultTuple> iData = ListUtil.shuffle(DataSource.getIrisData(), random);
//
//        int div = ((Double) (iData.size() * 0.3)).intValue();
//        List<DataResultTuple> trainSet = iData.subSequence(div);
//
//
//        List.range(0, 100).map(v -> ListUtil.shuffle(trainSet, random)).forEach(ts -> ts.forEach(wd -> {
//            Input input = new Input(wd.getData()).getNormalized();
//            network.propagate(input);
//
//            network.backPropagate(new Input(wd.getClassificationResult(categoryValues)));
//            network.updateWeights();
//
//        }));
//
//        List<DataResultTuple> testSet = iData.subSequence(0, div);
//        AtomicInteger valid = new AtomicInteger();
//        testSet.forEach(wd -> {
//
//            network.propagate(new Input(wd.getData()).getNormalized());
//            Input result = network.getResult();
//
//            String correct = "_";
//            if (result.getValues().indexOf(result.getValues().max().get()) == wd.getClassificationResult(categoryValues).indexOf(wd.getClassificationResult(categoryValues).max().get())) {
//                valid.getAndIncrement();
//                correct = "OK";
//            }
//            System.out.println(
//                    "shouldBe: " + wd.getClassificationResult(categoryValues)
//                            + " is: " + result.getValues().map(v -> FormatUtil.DECIMAL_FORMAT.format(v))
//                            + " " + correct
//            );
//        });
//
//        double result = ((double) valid.get() / (double) testSet.size());
//        System.out.println("r: " + result);
//        assertEquals(0.9555555555555556, result, 0.001);
//    }

}
