package com.example.drawmnist.draw_mnist;

import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import me.panavtec.drawableview.DrawableView;
import me.panavtec.drawableview.DrawableViewConfig;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;


public class MainActivity extends AppCompatActivity {
    private DrawableView drawableView;
    private DrawableViewConfig config;
    private ImageView imageView;
    private Interpreter tflite;

    private int[] intValues = new int[28 * 28];
    protected ByteBuffer imgData = null;
    private float[][] labelProbArray = null;

    protected String getModelPath() {
        return "model.tflite";
    }

    /** Writes Image data into a {@code ByteBuffer}. */
    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }
        imgData.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // Convert the image to floating point.
        int pixel = 0;
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                final int val = intValues[pixel++];
                addPixelValue(val);
            }
        }
    }

    protected void addPixelValue(int pixelValue) {
        // FIXME.
        imgData.putFloat((float) ((pixelValue >> 23) & 0xFF) / 255);
    }

    /** Memory-map the model file in Assets. */
    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd(getModelPath());
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private String buildResultText(float[] probs) {
        StringBuilder sb = new StringBuilder();
        for (int i=0; i<10; i++) {
            sb.append(String.format("%d[%.2f]\t", i, Array.get(probs, i)));
        }
        return sb.toString();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        try {
            tflite = new Interpreter(loadModelFile());
        } catch (IOException e) {
            Log.d(">>>>>>> ", "Failed to load model file.");
        }

        imgData = ByteBuffer.allocateDirect(1 * 28 * 28 * 1 * 4);
        imgData.order(ByteOrder.nativeOrder());
        labelProbArray = new float[1][10];

        drawableView = findViewById(R.id.paintView);
        Button clearButton = findViewById(R.id.clearButton);
        Button detectButton = findViewById(R.id.detectButton);
        imageView = findViewById(R.id.imageView);

        config = new DrawableViewConfig();
        config.setStrokeColor(getResources().getColor(android.R.color.black));
        config.setShowCanvasBounds(true); // If the view is bigger than canvas, with this the user will see the bounds (Recommended)
        config.setStrokeWidth(50.0f);
        config.setMinZoom(1.0f);
        config.setMaxZoom(1.0f);
        config.setCanvasHeight(1120);
        config.setCanvasWidth(1120);
        drawableView.setConfig(config);

        clearButton.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) {
                drawableView.clear();
                imageView.setImageResource(0);
            }
        });

        detectButton.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) {
                Bitmap bitmap = drawableView.obtainBitmap(Bitmap.createBitmap(config.getCanvasWidth(), config.getCanvasHeight(), Bitmap.Config.ALPHA_8));
                Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 28, 28, true);
                imageView.setImageBitmap(resizedBitmap);

                //calling tflite.run
                convertBitmapToByteBuffer(resizedBitmap);
                tflite.run(imgData, labelProbArray);
                String resultText = buildResultText(labelProbArray[0]);
                ((TextView) findViewById(R.id.resultTextView)).setText(resultText);
            }
        });
    }

}
