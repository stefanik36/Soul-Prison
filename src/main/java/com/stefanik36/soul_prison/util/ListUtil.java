package com.stefanik36.soul_prison.util;

import io.vavr.collection.List;

import java.util.Collections;
import java.util.Random;

public class ListUtil {


    public static <T> List<T> shuffle(List<T> list, Random random) {
        java.util.List<T> jList = list.toJavaList();
        Collections.shuffle(jList, random);
        return List.ofAll(jList);
    }

}
