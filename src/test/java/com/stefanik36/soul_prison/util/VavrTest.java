package com.stefanik36.soul_prison.util;

import io.vavr.collection.List;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class VavrTest {

    @Test
    public void dropRight() {
        List<Double> dl = List.of(1.0, 2.0, 3.0, 4.0).dropRight(1);

         assertEquals(dl.size(),3);
         assertEquals(dl.get(0),1.0,0.0001);
         assertEquals(dl.get(1),2.0,0.0001);
         assertEquals(dl.get(2),3.0,0.0001);
    }


    @Test
    public void foldLeft() {
        String resultFl = List.of("a", "b", "c").foldLeft("!", (xs, x) -> xs + x);
        String resultFr = List.of("a", "b", "c").reverse().foldRight("!", (x, xs) -> x + xs);

        assertEquals(resultFl,"!abc");
        assertEquals(resultFr,"cba!");
    }


}
