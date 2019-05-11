package com.stefanik36.soul_prison.util;

import io.vavr.collection.List;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.*;

public class ListUtilTest {

    @Test
    public void shuffle() {
        List<Integer> rList = ListUtil.shuffle(List.of(1, 2, 3, 4), new Random(33));

        assertEquals(rList.size(),4);
        assertEquals(rList.get(0).intValue(),4);
        assertEquals(rList.get(1).intValue(),2);
        assertEquals(rList.get(2).intValue(),1);
        assertEquals(rList.get(3).intValue(),3);
    }
}