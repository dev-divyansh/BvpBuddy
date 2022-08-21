package com.divyansh.bvpbuddy;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.divyansh.bvpbuddy.ml.ModelUnquant;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    protected Button button;
    protected ImageView res_image;
    protected TextView res_text;
    int imagesize = 224;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        button = (Button)findViewById(R.id.result);
        res_image = (ImageView)findViewById(R.id.imageView);
        res_text = (TextView)findViewById(R.id.result_text);

        if(ContextCompat.checkSelfPermission(MainActivity.this,
                Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED){
            ActivityCompat.requestPermissions(MainActivity.this,
                    new String[]{Manifest.permission.CAMERA} , 101);
        }

        button.setOnClickListener(new View.OnClickListener(){

            @Override
            public void onClick(View view) {
                Intent  intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(intent , 101);
            }
        });

    }
    public void classifyImage(Bitmap bitmap){
        try {
            ModelUnquant model = ModelUnquant.newInstance(getApplicationContext());

            // Creates inputs for reference.
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4*imagesize*imagesize*3);
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);

            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imagesize*imagesize];
            bitmap.getPixels(intValues , 0 , bitmap.getWidth() , 0 ,0,bitmap.getWidth() , bitmap.getHeight());
            int pixel = 0;
            for(int i =0 ; i<imagesize ;i++){
                for(int j =0; j<imagesize ;j++){
                    int val = intValues[pixel++];
                    byteBuffer.putFloat(((val>>16)& 0xFF)*(1.f / 1));
                    byteBuffer.putFloat(((val>>8)& 0xFF)*(1.f / 1));
                    byteBuffer.putFloat((val & 0xFF)*(1.f / 1));
                }
            }



            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            ModelUnquant.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            // finding class with highest order of confidence
            float[] confidences = outputFeature0.getFloatArray();

            int maxPos = 0;
            float maxConfidence = 0;
            for(int i = 0;i<confidences.length;i++){
                if(confidences[i] > maxConfidence){
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"CE" , "IT"};
            res_text.setText(classes[maxPos]);

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if(requestCode == 101){
            Bitmap bitmap = (Bitmap) data.getExtras().get("data");
            int dimension = Math.min(bitmap.getWidth() , bitmap.getHeight());
            bitmap = ThumbnailUtils.extractThumbnail(bitmap , dimension , dimension);
            res_image.setImageBitmap(bitmap);

            bitmap = Bitmap.createScaledBitmap(bitmap , imagesize , imagesize , false);

            classifyImage(bitmap);
        }
    }
}