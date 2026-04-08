package com.surendramaran.yolov8tflite

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.view.View
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.surendramaran.yolov8tflite.Constants.LABELS_PATH
import com.surendramaran.yolov8tflite.Constants.MODEL_PATH
import com.surendramaran.yolov8tflite.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), Detector.DetectorListener, DepthEstimator.DepthListener {
    private lateinit var binding: ActivityMainBinding
    private val isFrontCamera = false

    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var cameraProvider: ProcessCameraProvider? = null

    private var detector: Detector? = null
    private var depthEstimator: DepthEstimator? = null

    private var frameCounter = 0
    private lateinit var cameraExecutor: ExecutorService

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()

        // Sync-safe Initialization: Wait for models to load before starting camera
        cameraExecutor.execute {
            try {
                // Initialize YOLO
                detector = Detector(baseContext, MODEL_PATH, LABELS_PATH, this)

                // Initialize MiDaS (Ensure this file is in your assets folder)
                depthEstimator = DepthEstimator(baseContext, "midas_model.tflite", this)

                // Only start camera on UI thread once models are ready
                runOnUiThread {
                    if (allPermissionsGranted()) {
                        startCamera()
                    } else {
                        requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "FATAL: Model loading failed. Check assets! ${e.message}")
            }
        }

        bindListeners()
    }

    private fun bindListeners() {
        binding.isGpu.setOnCheckedChangeListener { buttonView, isChecked ->
            cameraExecutor.submit {
                detector?.restart(isGpu = isChecked)
            }
            val color = if (isChecked) R.color.orange else R.color.gray
            buttonView.setBackgroundColor(ContextCompat.getColor(baseContext, color))
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: return
        val rotation = binding.viewFinder.display.rotation

        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(rotation)
            .build()

        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetRotation(rotation)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        imageAnalyzer?.setAnalyzer(cameraExecutor) { imageProxy ->
            // Safety Guard: If models aren't ready yet or are closing, skip frame
            val currentDetector = detector
            val currentDepth = depthEstimator

            if (currentDetector == null || currentDepth == null) {
                imageProxy.close()
                return@setAnalyzer
            }

            val bitmapBuffer = Bitmap.createBitmap(imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888)
            imageProxy.use { bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer) }

            val matrix = Matrix().apply {
                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
            }

            val rotatedBitmap = Bitmap.createBitmap(bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height, matrix, true)

            // 1. YOLO Detection (Every frame)
            currentDetector.detect(rotatedBitmap)

            // 2. MiDaS Depth (Every 5th frame to maintain high FPS)
            if (frameCounter % 5 == 0) {
                currentDepth.estimate(rotatedBitmap)
            }
            frameCounter++

            imageProxy.close()
        }

        cameraProvider.unbindAll()
        try {
            cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
            preview?.setSurfaceProvider(binding.viewFinder.surfaceProvider)
        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    // --- YOLO Callbacks ---
    override fun onEmptyDetect() {
        runOnUiThread { binding.overlay.clear() }
    }

    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long) {
        runOnUiThread {
            binding.inferenceTime.text = "${inferenceTime}ms"
            binding.overlay.apply {
                setResults(boundingBoxes)
                invalidate()
            }
        }
    }

    // --- MiDaS Depth Callback ---
    override fun onDepthMapGenerated(depthMap: Array<FloatArray>, inferenceTime: Long) {
        // Safety check to ensure activity is still active
        if (isFinishing || isDestroyed) return

        val size = depthMap.size
        val bitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)

        // Find min/max for normalization (Heatmap Contrast)
        var minVal = Float.MAX_VALUE
        var maxVal = Float.MIN_VALUE
        for (row in depthMap) {
            for (v in row) {
                if (v < minVal) minVal = v
                if (v > maxVal) maxVal = v
            }
        }
        val range = maxVal - minVal

        for (y in 0 until size) {
            for (x in 0 until size) {
                val normalized = if (range > 0) {
                    (((depthMap[y][x] - minVal) / range) * 255).toInt().coerceIn(0, 255)
                } else 0

                // Heatmap: Red (Close) to Blue (Far)
                bitmap.setPixel(x, y, Color.rgb(normalized, 0, 255 - normalized))
            }
        }

        runOnUiThread {
            binding.depthOverlay.visibility = View.VISIBLE
            binding.depthOverlay.setImageBitmap(bitmap)
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private val requestPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
        if (it[Manifest.permission.CAMERA] == true) startCamera()
    }

    override fun onDestroy() {
        super.onDestroy()
        detector?.close()
        depthEstimator?.close()
        cameraExecutor.shutdown()
    }

    override fun onResume() {
        super.onResume()
        // If models are ready, start camera. If not, onCreate's executor handles it.
        if (allPermissionsGranted() && detector != null && depthEstimator != null) {
            startCamera()
        }
    }

    companion object {
        private const val TAG = "Camera"
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}