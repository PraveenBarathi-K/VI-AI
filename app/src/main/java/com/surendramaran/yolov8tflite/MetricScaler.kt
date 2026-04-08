package com.surendramaran.yolov8tflite

import kotlin.math.sqrt

class MetricScaler {
    private var lastTimestamp = 0L
    private val alpha = 0.95f // Smoothing filter
    private var filteredAccel = 0f

    fun getStableScale(accel: FloatArray, timestamp: Long): Float {
        if (lastTimestamp == 0L) {
            lastTimestamp = timestamp
            return 1.5f // Start with a default 1.5m scale
        }

        val linearAccel = sqrt(accel[0]*accel[0] + accel[1]*accel[1] + accel[2]*accel[2])

        // Low-pass filter to remove jitter
        filteredAccel = alpha * filteredAccel + (1 - alpha) * linearAccel

        lastTimestamp = timestamp

        // Scale clamping: ensure the multiplier stays within realistic human bounds (0.5m to 5.0m)
        // This prevents the "1000 meters" error
        return (filteredAccel * 2.0f).coerceIn(1.0f, 5.0f)
    }
}