package ai.jhu.edu.flfm;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

public class MyTranslator implements Translator<NDArray[], NDArray> {
    @Override
    public NDArray processOutput(TranslatorContext ctx, NDList list) {
        return list.get(0);
    }

    @Override
    public NDList processInput(TranslatorContext ctx, NDArray[] input) {
        NDArray img = input[0];
        NDArray psf = input[1];

        return new NDList(img, psf);
    }
}

