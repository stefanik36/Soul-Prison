package com.stefanik36.soul_prison.data;

import io.vavr.collection.List;
import org.junit.Test;

import static org.junit.Assert.*;

public class DataResultTupleTest {

    @Test
    public void getClassificationResult() {
        DataResultTuple dataResultTuple = new DataResultTuple(1.1, List.of(2.3,4.2,5.6));

        List<Double> result = dataResultTuple.getClassificationResult(List.of(0.2, 1.1, 5.5));

        assertEquals(result.get(0),0.0,0.000001);
        assertEquals(result.get(1),1.0,0.000001);
        assertEquals(result.get(2),0.0,0.000001);
    }
}