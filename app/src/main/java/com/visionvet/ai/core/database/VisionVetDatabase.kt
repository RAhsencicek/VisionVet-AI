package com.visionvet.ai.core.database

import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import android.content.Context
import com.visionvet.ai.core.database.dao.UserDao
import com.visionvet.ai.core.database.dao.AnalysisDao
import com.visionvet.ai.core.database.dao.BacterialResultDao
import com.visionvet.ai.core.database.model.User
import com.visionvet.ai.core.database.model.Analysis
import com.visionvet.ai.core.database.model.BacterialResult

@Database(
    entities = [User::class, Analysis::class, BacterialResult::class],
    version = 3,
    exportSchema = false
)
abstract class VisionVetDatabase : RoomDatabase() {
    abstract fun userDao(): UserDao
    abstract fun analysisDao(): AnalysisDao
    abstract fun bacterialResultDao(): BacterialResultDao

    companion object {
        @Volatile
        private var INSTANCE: VisionVetDatabase? = null

        fun getDatabase(context: Context): VisionVetDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    VisionVetDatabase::class.java,
                    "visionvet_database"
                )
                .fallbackToDestructiveMigration() // For development - removes old data
                .build()
                INSTANCE = instance
                instance
            }
        }
    }
}
