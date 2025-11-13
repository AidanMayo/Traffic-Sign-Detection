# Traffic Sign Recognition App

Basic Android app for real-time traffic sign recognition using camera feed.

## Project Overview

This is a minimal Android application built with Kotlin that provides a foundation for traffic sign recognition. The app displays a live camera feed and is ready for ML model integration to detect and classify traffic signs in real-time.

## Features

- **Live Camera Preview**: Real-time camera feed using CameraX

- **Detection Overlay**: Custom view for drawing bounding boxes and labels

- **Text-to-Speech**: Voice announcements for detected signs

- **Navy Blue Theme**: Modern navy blue color scheme

- **ML-Ready Architecture**: Repository pattern ready for model integration

## Project Structure

### Source Files (`app/src/main/java/com/example/tsrapp/`)

#### Application

- **`TSRApplication.kt`**: Application class for app-wide initialization

#### Data Layer

- **`data/model/TrafficSign.kt`**: Data model for detected traffic signs

  - Contains sign label, confidence, bounding box coordinates, and critical flag

- **`data/repository/TSRRepository.kt`**: Repository for ML model integration

  - `detectSignsInFrame(bitmap: Bitmap)`: Main method to implement ML inference

#### UI Layer

- **`ui/main/MainActivity.kt`**: Main activity with camera integration

  - Handles camera permissions

  - Processes camera frames

  - Manages detection overlay and TTS announcements

- **`ui/main/MainViewModel.kt`**: ViewModel for managing detection state

  - Processes frames through repository

  - Exposes detected signs via LiveData

- **`ui/main/DetectionOverlayView.kt`**: Custom view for drawing detections

  - Draws bounding boxes (red for critical, green for normal)

  - Displays labels with confidence percentages

#### Utilities

- **`util/TextToSpeechHelper.kt`**: Helper class for voice announcements

  - Initializes TTS engine

  - Speaks detected sign labels

### Resource Files (`app/src/main/res/`)

#### Layouts

- **`layout/activity_main.xml`**: Main activity layout

  - Camera preview view

  - Detection overlay view

#### Values

- **`values/colors.xml`**: Color definitions

  - Navy blue primary colors (`navy_500`, `navy_700`)

  - Blue secondary colors (`blue_200`, `blue_700`)

- **`values/strings.xml`**: String resources

  - App name, permission messages

- **`values/themes.xml`**: App theme configuration

  - Material Design theme with navy blue color scheme

#### Drawables

- **`drawable/ic_launcher_foreground.xml`**: Launcher icon foreground

#### Mipmaps

- **`mipmap-anydpi-v26/ic_launcher.xml`**: Adaptive launcher icon

- **`mipmap-anydpi-v26/ic_launcher_round.xml`**: Round launcher icon

#### XML Configuration

- **`xml/backup_rules.xml`**: Backup configuration

- **`xml/data_extraction_rules.xml`**: Data extraction rules

### Configuration Files

#### Root Level

- **`build.gradle.kts`**: Project-level build configuration

- **`settings.gradle.kts`**: Project settings and module inclusion

- **`gradle.properties`**: Gradle configuration properties

- **`gradlew`**: Gradle wrapper script

- **`gradle/wrapper/`**: Gradle wrapper files

#### App Level

- **`app/build.gradle.kts`**: App-level build configuration

  - Dependencies: CameraX, Lifecycle, Material Components

  - View binding enabled

- **`app/proguard-rules.pro`**: ProGuard rules for release builds

- **`app/src/main/AndroidManifest.xml`**: App manifest

  - Camera permission

  - Main activity declaration

## What Was Implemented

### Core Functionality

1. **Camera Integration**

   - CameraX library for camera access

   - Real-time frame processing pipeline

   - ImageProxy to Bitmap conversion for ML processing

2. **UI Components**

   - Camera preview using PreviewView

   - Custom detection overlay view

   - Material Design components

3. **Architecture**

   - MVVM pattern with ViewModel

   - Repository pattern for data layer

   - LiveData for reactive updates

4. **Features**

   - Camera permission handling

   - Text-to-speech initialization

   - Detection visualization (ready for ML output)

### Design Decisions

- **Navy Blue Theme**: Professional color scheme replacing default purple

- **Minimal UI**: Clean, simple interface focused on camera and detections

- **Offline-First**: Designed for on-device ML model (no backend required)

- **Real-Time Processing**: Optimized for frame-by-frame detection

## Dependencies

- **CameraX**: Camera functionality (`1.3.1`)

- **Lifecycle**: ViewModel and LiveData (`2.7.0`)

- **Material Components**: UI components (`1.11.0`)

- **Core KTX**: Kotlin extensions (`1.12.0`)

- **AppCompat**: Backward compatibility (`1.6.1`)

## Setup

1. Open project in Android Studio

2. Sync Gradle files (File → Sync Project with Gradle Files)

3. Connect Android device or start emulator

4. Run the app (Run → Run 'app' or Shift+F10)

## ML Model Integration

To integrate your ML model:

1. Add your model file (`.tflite` or similar) to `app/src/main/assets/`

2. Update `TSRRepository.kt`:

   ```kotlin
   suspend fun detectSignsInFrame(bitmap: Bitmap): List<TrafficSign> {
       // Load and run your ML model here
       // Convert model output to TrafficSign objects
       return detectedSigns
   }
   ```

3. The app will automatically display detections with bounding boxes and labels

## Permissions

- `CAMERA`: Required for live camera feed

## Build Configuration

- **Min SDK**: 24 (Android 7.0)

- **Target SDK**: 34 (Android 14)

- **Compile SDK**: 34

- **Kotlin**: 1.9.20

- **Gradle**: 8.2

## Notes

- The app currently returns empty detections until ML model is integrated

- Camera preview works on both physical devices and emulators

- Text-to-speech uses device's default language

- All UI elements use navy blue color scheme
