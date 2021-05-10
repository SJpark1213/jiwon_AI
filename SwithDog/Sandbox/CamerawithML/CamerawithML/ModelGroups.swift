//
//  ModelGroups.swift
//  CamerawithML
//
//  Created by 강지원 on 2020/10/26.
//

import Foundation

struct Exercise{
    var numberOfReps = 0
    var probability = 0.0
}

enum Exercises{
    case squat, lunge, jumpingJack, standStill, wrongMove
}

var squat = Exercise(numberOfReps: 0, probability: 0)
var lunge = Exercise(numberOfReps: 0, probability: 0)
var jumpingJack = Exercise(numberOfReps: 0, probability: 0)
var standStill = Exercise(numberOfReps: 0, probability: 0)

var exerciseList : [String: Exercise] = ["Squat": squat, "Lunge": lunge, "JumpingJack": jumpingJack, "StandStill": standStill]
