package ai.jhu.edu.flfm;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

public class MyTranslator implements Translator<Image[], Image[]> {
    @Override
    public Image[] processOutput(TranslatorContext ctx, NDList list) {
        System.out.println("Output shape: " + list.get(0).getShape());
        return new Image[]{ImageFactory.getInstance().fromNDArray(list.get(0))};
    }

    @Override
    public NDList processInput(TranslatorContext ctx, Image[] input) {
        NDArray img = input[0].toNDArray(ctx.getNDManager()).transpose(2, 0, 1);
        NDArray psf = input[1].toNDArray(ctx.getNDManager()).transpose(2, 0, 1);
        System.out.println(img.getShape());
        System.out.println(psf.getShape());

        return new NDList(
            img,
            psf
        );
    }
}

