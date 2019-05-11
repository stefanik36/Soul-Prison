package com.stefanik36.soul_prison.data;

import io.vavr.collection.List;
import org.junit.Test;

import static org.junit.Assert.*;

public class InputTest {

    @Test
    public void getNormalized() {

        Input input = new Input(List.of(1.0, 2.0, 3.0));

        Input normalized =  input.getNormalized();

        assertEquals(normalized.getValues().size(),3);
        assertEquals(normalized.getValues().get(0),0.0,0.000001);
        assertEquals(normalized.getValues().get(1),0.5,0.000001);
        assertEquals(normalized.getValues().get(2),1.0,0.000001);

    }
}