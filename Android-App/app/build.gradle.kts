    plugins {
        alias(libs.plugins.android.application)
        alias(libs.plugins.kotlin.android)
        alias(libs.plugins.kotlin.compose)
    }
    configurations.all {
        resolutionStrategy {
            // Force the newer annotations version
            force("org.jetbrains:annotations:23.0.0")
            // Remove the old annotations completely
            exclude(group = "com.intellij", module = "annotations")
        }
    }


    android {
        namespace = "com.example.sm"
        compileSdk = 35

        defaultConfig {
            applicationId = "com.example.sm"
            minSdk = 24
            targetSdk = 35
            versionCode = 1
            versionName = "1.0"

            testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"


            buildConfigField(
                "String",
                "GEMINI_API_KEY",
                "\"${project.properties["AIzaSyDfd55ooqV5JU6Wnvxwtk-c7_s7e_ZtldE"]}\""
            )
        }

        buildTypes {
            release {
                isMinifyEnabled = false
                proguardFiles(
                    getDefaultProguardFile("proguard-android-optimize.txt"),
                    "proguard-rules.pro"
                )
            }
        }

        compileOptions {
            sourceCompatibility = JavaVersion.VERSION_11
            targetCompatibility = JavaVersion.VERSION_11
        }

        kotlinOptions {
            jvmTarget = "11"
        }

        buildFeatures {
            compose = true
            buildConfig = true
        }
        applicationVariants.all {
            outputs.all {
                if (name == "debug") {
                    (this as com.android.build.gradle.internal.api.BaseVariantOutputImpl).outputFileName =
                        "Score Matrix.apk"
                }
            }
        }
    }

    dependencies {

        implementation("org.jetbrains:annotations:23.0.0") {
            exclude(group = "com.intellij", module = "annotations")
        }
        // Navigation
        implementation("androidx.navigation:navigation-compose:2.7.6")

        // Compose Material3
        implementation("androidx.compose.material3:material3:1.2.1")

        // ML Kit
        implementation("com.google.android.gms:play-services-mlkit-document-scanner:16.0.0-beta1")

        // PDF
        implementation("com.itextpdf:itext7-core:7.2.3")

        // Gemini AI
        implementation("com.google.ai.client.generativeai:generativeai:0.3.0")

        // Protocol Buffers
        implementation("com.google.protobuf:protobuf-javalite:3.25.1")

        // EXIF
        implementation("androidx.exifinterface:exifinterface:1.3.6")

        // Compose Activity
        implementation("androidx.activity:activity-compose:1.8.2")

        // Coil Image Loading
        implementation("io.coil-kt:coil-compose:2.5.0")
        implementation("com.github.jeziellago:compose-markdown:0.3.2")


        // Compose BOM
        implementation(platform(libs.androidx.compose.bom))
        implementation(libs.androidx.ui)
        implementation(libs.androidx.ui.graphics)
        implementation(libs.androidx.ui.tooling.preview)
        implementation(libs.androidx.material3)
        implementation(libs.androidx.navigation.runtime.android)

        // Kotlin Extensions
        implementation(libs.androidx.core.ktx)
        implementation(libs.androidx.lifecycle.runtime.ktx)
        implementation(libs.androidx.activity.compose)

        // Networking & JSON
        implementation("com.squareup.okhttp3:okhttp:4.12.0")
        implementation("org.json:json:20231013")
        implementation(libs.androidx.room.common.jvm)
        implementation(libs.androidx.room.compiler)
        implementation(libs.androidx.room.runtime.android)

        // Testing
        testImplementation(libs.junit)
        androidTestImplementation(libs.androidx.junit)
        androidTestImplementation(libs.androidx.espresso.core)
        androidTestImplementation(platform(libs.androidx.compose.bom))
        androidTestImplementation(libs.androidx.ui.test.junit4)
        debugImplementation(libs.androidx.ui.tooling)
        debugImplementation(libs.androidx.ui.test.manifest)
        implementation("org.jetbrains:annotations:23.0.0")
    }
