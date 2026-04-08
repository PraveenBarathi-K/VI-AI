package com.surendramaran.yolov8tflite

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder

class DepthEstimator(
    context: Context,
    modelPath: String,
    private val listener: DepthListener
) {
    private var interpreter: Interpreter? = null
    private val inputSize = 256 // MiDaS Small standard

    interface DepthListener {
        fun onDepthMapGenerated(depthMap: Array<FloatArray>, inferenceTime: Long)
    }

    init {
        try {
            val options = Interpreter.Options().apply {
                addDelegate(org.tensorflow.lite.gpu.GpuDelegate())
            }
            interpreter = Interpreter(FileUtil.loadMappedFile(context, modelPath), options)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun estimate(bitmap: Bitmap) {
        val startTime = SystemClock.uptimeMillis()

        // 1. Pre-process the image
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val inputBuffer = convertBitmapToBuffer(resizedBitmap)

        // 2. Prepare the output buffer to match [1, 256, 256, 1]
        val outputBuffer = Array(1) { Array(inputSize) { Array(inputSize) { FloatArray(1) } } }

        // 3. Run Inference
        try {
            interpreter?.run(inputBuffer, outputBuffer)
        } catch (e: Exception) {
            e.printStackTrace()
            return
        }

        val endTime = SystemClock.uptimeMillis()

        // 4. Flatten [1][256][256][1] to [256][256] for the listener
        val flattenedOutput = Array(inputSize) { y ->
            FloatArray(inputSize) { x ->
                outputBuffer[0][y][x][0]
            }
        }

        listener.onDepthMapGenerated(flattenedOutput, endTime - startTime)
    }

    private fun convertBitmapToBuffer(bitmap: Bitmap): ByteBuffer {
        val imgData = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4)
        imgData.order(ByteOrder.nativeOrder())

        val intValues = IntArray(inputSize * inputSize)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        imgData.rewind()
        for (pixelValue in intValues) {
            // Normalize to [0, 1] as expected by MiDaS
            imgData.putFloat((pixelValue shr 16 and 0xFF) / 255.0f)
            imgData.putFloat((pixelValue shr 8 and 0xFF) / 255.0f)
            imgData.putFloat((pixelValue and 0xFF) / 255.0f)
        }
        return imgData
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}