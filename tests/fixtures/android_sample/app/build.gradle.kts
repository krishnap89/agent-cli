plugins {
    id("com.android.application")
    kotlin("android")
}

android {
    namespace = "com.example.app"
}

dependencies {
    implementation(project(":core:data"))
    // implementation(project(":ghost"))  commented out: must be ignored
    testImplementation("junit:junit:4.13.2")
}
