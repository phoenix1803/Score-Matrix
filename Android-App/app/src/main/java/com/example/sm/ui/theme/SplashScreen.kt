package com.example.sm

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import androidx.activity.ComponentActivity



import androidx.activity.compose.setContent
import androidx.compose.animation.core.*

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.scale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import com.example.sm.ui.theme.SMTheme

class SplashScreen : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)


        Handler(Looper.getMainLooper()).postDelayed({
            startActivity(Intent(this, MainActivity::class.java))
            finish()
        }, 1000)

        setContent {
            SMTheme  {
                SplashScreenUI()
            }
        }
    }
}

@Composable
fun SplashScreenUI() {
    Surface(
        modifier = Modifier.fillMaxSize(),
        color = MaterialTheme.colorScheme.background
    ) {
        Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center
        ) {
            ZoomInLogo()
        }
    }
}

@Composable
fun ZoomInLogo() {
    var startAnimation by remember { mutableStateOf(false) }

    val scale by animateFloatAsState(
        targetValue = if (startAnimation) 1.0f else 0.5f,
        animationSpec = tween(durationMillis = 500, easing = FastOutSlowInEasing),
        label = "Zoom Animation"
    )

    LaunchedEffect(Unit) {
        startAnimation = true
    }

    Image(
        painter = painterResource(id = R.drawable.logo),
        contentDescription = "App Logo",
        modifier = Modifier
            .scale(scale)
            .fillMaxWidth(1f)
            .aspectRatio(0.01f),
    )
}
