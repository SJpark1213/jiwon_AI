//
//  ModelsCollection.swift
//  Pages
//
//  Created by 강지원 on 2020/11/14.
//

import Foundation
import UIKit
import CoreData

// Dog model
struct Dog{
    var serialNumber : Int
    var dogBreed : Int
    var dogName : String
    
    var health : Double
    var intimacy : Double
    var hunger : Double
    
    var hapiness : Double
    
    init(dogName: String, serialNumber: Int, dogBreed: Int) {
        self.dogName = dogName
        self.dogBreed = dogBreed
        self.serialNumber = serialNumber
        self.health = 0.0
        self.intimacy = 0.0
        self.hunger = 0.0
        self.hapiness = 0.0
    }
}

let dogBreedList = ["Practice","Colly","Bishong","Pome","York","Mix"]

struct Exercise{
    let exerciseType : String
    var numberOfReps = 0
    var probability = 0.0
    
    init(_ exerciseType: String) {
        self.exerciseType = exerciseType
    }
}

var squat = Exercise("squat")
var lunge = Exercise("lunge")
var jumpingJack = Exercise("jumpingJack")
var standStill = Exercise("standStill")

//let exerciseList : [String: Exercise] = ["Squat": squat, "Lunge": lunge, "JumpingJack": jumpingJack, "StandStill": standStill]

struct Plan{
    var planName : String
    var isPaidPlan : Bool
    var price : Double
    var numberOfDays : Int
    var squatRepsPerDay : Int
    var lungeRepsPerDay : Int
    var jumpingJackRepsPerDay : Int
    var planImage : UIImage
    
    init(planName: String, isPaidPlan: Bool, price: Double, numberOfDays: Int, squatRepsPerDay: Int, lungeRepsPerDay: Int, jumpingJackRepsPerDay: Int, planImage: UIImage) {
        self.planName = planName
        self.isPaidPlan = isPaidPlan
        self.price = price
        self.numberOfDays = numberOfDays
        self.squatRepsPerDay = squatRepsPerDay
        self.lungeRepsPerDay = lungeRepsPerDay
        self.jumpingJackRepsPerDay = jumpingJackRepsPerDay
        self.planImage = planImage
    }
}

let freePlan = Plan(planName: "무료 플랜", isPaidPlan: false, price: 0, numberOfDays: 0, squatRepsPerDay: 50, lungeRepsPerDay: 50, jumpingJackRepsPerDay: 50, planImage: #imageLiteral(resourceName: "free plan icon"))

let PlanA = Plan(planName: "7일 플랜",isPaidPlan: true, price: 5, numberOfDays: 7, squatRepsPerDay: 50, lungeRepsPerDay: 50, jumpingJackRepsPerDay: 50, planImage: #imageLiteral(resourceName: "brass coin"))

let PlanB = Plan(planName: "14일 플랜",isPaidPlan: true, price: 10, numberOfDays: 14, squatRepsPerDay: 100, lungeRepsPerDay: 100, jumpingJackRepsPerDay: 100, planImage: #imageLiteral(resourceName: "brass medal"))

let PlanC = Plan(planName: "30일 플랜",isPaidPlan: true, price: 20, numberOfDays: 21, squatRepsPerDay: 150, lungeRepsPerDay: 150, jumpingJackRepsPerDay: 150, planImage: #imageLiteral(resourceName: "silver medal"))

let PlanD = Plan(planName: "60일 플랜",isPaidPlan: true, price: 40, numberOfDays: 28, squatRepsPerDay: 200, lungeRepsPerDay: 200, jumpingJackRepsPerDay: 200, planImage: #imageLiteral(resourceName: "gold medal"))

let planList = [freePlan, PlanA, PlanB, PlanC, PlanD]



// Core Data, Global Variables
struct AppUser {
    var currentDogSerialNumber: Int = 0
    var currentPlan: Int = 0
    var currentPlanAchievementRate: Double = 0.0
    var currentPlanDaysList: [String] = []
    
    var todaySquatReps: Int = 0
    var todayLungeReps: Int = 0
    var todayJumpingJackReps: Int = 0
    var todayAchievementRate: Double = 0.0
    
    var graduateDogSerialList: [Int] = []
    var graduatePlanList: [Int] = [0]
    var graduatePlanAchievementRates: [Double] = []
    
    var runAwayDogSerialList: [Int] = []
    var runAwayPlanList: [Int] = []
    var runAwayPlanAchievementRates: [Double] = []
    
    var pastSquatRepsDict: [String: Int] = [:]
    var pastLungeRepsDict: [String: Int] = [:]
    var pastJumpingJackRepsDict: [String: Int] = [:]
}

var nowUser = AppUser()

func addUser(serialNum: Int, plan: Int){
    guard let appDelegate = UIApplication.shared.delegate as? AppDelegate else {
        return
    }
    
    let managedContext = appDelegate.persistentContainer.viewContext
            
    let entity = NSEntityDescription.entity(forEntityName: "User", in: managedContext)!
    let newUser = NSManagedObject(entity: entity, insertInto: managedContext)
    
    newUser.setValue(serialNum, forKeyPath: "currentDogSerialNumber")
    newUser.setValue(plan, forKeyPath: "currentPlan")
    newUser.setValue(0, forKeyPath: "currentPlanAchievementRate")
    newUser.setValue([], forKey: "currentPlanDaysList")
    newUser.setValue(0, forKeyPath: "todaySquatReps")
    newUser.setValue(0, forKeyPath: "todayLungeReps")
    newUser.setValue(0, forKeyPath: "todayJumpingJackReps")
    newUser.setValue(0, forKeyPath: "todayAchievementRate")
    newUser.setValue([], forKeyPath: "graduateDogSerialList")
    newUser.setValue([0], forKeyPath: "graduatePlanList")
    newUser.setValue([], forKeyPath: "graduatePlanAchievementRates")
    newUser.setValue([], forKeyPath: "runAwayDogSerialList")
    newUser.setValue([], forKeyPath: "runAwayPlanList")
    newUser.setValue([], forKeyPath: "runAwayPlanAchievementRates")
    newUser.setValue([:], forKey: "pastSquatRepsDict")
    newUser.setValue([:], forKey: "pastLungeRepsDict")
    newUser.setValue([:], forKey: "pastJumpingJackRepsDict")

    do {
        try managedContext.save()
    } catch let error as NSError {
        print("Could not save. \(error), \(error.userInfo)")
    }
}

func fetchUserData(){
    guard let appDelegate = UIApplication.shared.delegate as? AppDelegate else {
          return
    }
      
    let managedContext = appDelegate.persistentContainer.viewContext
    let fetchRequest = NSFetchRequest<NSManagedObject>(entityName: "User")
    
    var user = NSManagedObject()
    
    do {
        user = try managedContext.fetch(fetchRequest)[0]
    } catch let error as NSError {
        print("Could not fetch. \(error), \(error.userInfo)")
    }
    
    nowUser.currentDogSerialNumber = user.value(forKeyPath: "currentDogSerialNumber") as? Int ?? 0
    nowUser.currentPlan = user.value(forKeyPath: "currentPlan") as? Int ?? 0
    nowUser.currentPlanAchievementRate = user.value(forKeyPath: "currentPlanAchievementRate") as? Double ?? 0.0
    nowUser.currentPlanDaysList = user.value(forKeyPath: "currentPlanDaysList") as? [String] ?? []
    
    nowUser.todaySquatReps = user.value(forKeyPath: "todaySquatReps") as? Int ?? 0
    nowUser.todayLungeReps = user.value(forKeyPath: "todayLungeReps") as? Int ?? 0
    nowUser.todayJumpingJackReps = user.value(forKeyPath: "todayJumpingJackReps") as? Int ?? 0
    nowUser.todayAchievementRate = user.value(forKeyPath: "todayAchievementRate") as? Double ?? 0.0
    
    nowUser.graduateDogSerialList = user.value(forKeyPath: "graduateDogSerialList") as? [Int] ?? []
    nowUser.graduatePlanList = user.value(forKeyPath: "graduatePlanList") as? [Int] ?? []
    nowUser.graduatePlanAchievementRates = user.value(forKeyPath: "graduatePlanAchievementRates") as? [Double] ?? []
    
    nowUser.runAwayDogSerialList = user.value(forKeyPath: "runAwayDogSerialList") as? [Int] ?? []
    nowUser.runAwayPlanList = user.value(forKeyPath: "runAwayPlanList") as? [Int] ?? []
    nowUser.runAwayPlanAchievementRates = user.value(forKeyPath: "runAwayPlanAchievementRates") as? [Double] ?? []
    
    nowUser.pastSquatRepsDict = user.value(forKey: "pastSquatRepsDict") as? [String: Int] ?? [:]
    nowUser.pastLungeRepsDict = user.value(forKey: "pastLungeRepsDict") as? [String: Int] ?? [:]
    nowUser.pastJumpingJackRepsDict = user.value(forKey: "pastJumpingJackRepsDict") as? [String: Int] ?? [:]
}

func saveUserData(){
    guard let appDelegate = UIApplication.shared.delegate as? AppDelegate else {
          return
    }
      
    let managedContext = appDelegate.persistentContainer.viewContext
    let fetchRequest = NSFetchRequest<NSManagedObject>(entityName: "User")
    
    do {
        let user = try managedContext.fetch(fetchRequest)[0]
        user.setValue(nowUser.currentDogSerialNumber, forKey: "currentDogSerialNumber")
        user.setValue(nowUser.currentPlan, forKey: "currentPlan")
        user.setValue(nowUser.currentPlanAchievementRate, forKey: "currentPlanAchievementRate")
        user.setValue(nowUser.currentPlanDaysList, forKey: "currentPlanDaysList")
        user.setValue(nowUser.todaySquatReps, forKey: "todaySquatReps")
        user.setValue(nowUser.todayLungeReps, forKey: "todayLungeReps")
        user.setValue(nowUser.todayJumpingJackReps, forKey: "todayJumpingJackReps")
        user.setValue(nowUser.todayAchievementRate, forKey: "todayAchievementRate")
        user.setValue(nowUser.graduatePlanList, forKey: "graduatePlanList")
        user.setValue(nowUser.graduateDogSerialList, forKey: "graduateDogSerialList")
        user.setValue(nowUser.graduatePlanAchievementRates, forKey: "graduatePlanAchievementRates")
        user.setValue(nowUser.runAwayPlanList, forKey: "runAwayPlanList")
        user.setValue(nowUser.runAwayDogSerialList, forKey: "runAwayDogSerialList")
        user.setValue(nowUser.runAwayPlanAchievementRates, forKey: "runAwayPlanAchievementRates")
        user.setValue(nowUser.pastSquatRepsDict, forKey: "pastSquatRepsDict")
        user.setValue(nowUser.pastLungeRepsDict, forKey: "pastLungeRepsDict")
        user.setValue(nowUser.pastJumpingJackRepsDict, forKey: "pastJumpingJackRepsDict")

        try managedContext.save()
    }
    catch let error {
        print("edit User error :", error)
    }
}

func deleteUserData(){
    guard let appDelegate = UIApplication.shared.delegate as? AppDelegate else {
        return
    }
    
    let managedContext = appDelegate.persistentContainer.viewContext
    let fetchRequest = NSFetchRequest<NSFetchRequestResult>(entityName: "User")
    let batchDeleteRequest = NSBatchDeleteRequest(fetchRequest: fetchRequest)

    do {
        try managedContext.execute(batchDeleteRequest)

    } catch let error as NSError{
        print("Could not delete: ", error)
    }
}

// UserDefaults
func updateModel(){
    let userDefaults = UserDefaults.standard
    let totalDogNumbers = userDefaults.integer(forKey: "totalDogNumbers")
    
    if totalDogNumbers == 0{
        addUser(serialNum: 0, plan: 0)
        fetchUserData()
        userDefaults.setValue(1, forKey: "totalDogNumbers")
    }else{
        fetchUserData()
    }
}

func additionalFetch(){
    let formatter = DateFormatter()
    let stringToday = formatter.string(from: Date())
    formatter.dateFormat = "yyyy-MM-dd"
    if !nowUser.currentPlanDaysList.contains(stringToday){
        nowUser.currentPlan = 0
    }
}

let dogImageList: [UIImage] = [UIImage(named: "free plan dog")!,UIImage(named: "plan 1 dog")!,UIImage(named: "plan 2 dog")!,UIImage(named: "plan 3 dog")!,UIImage(named: "plan 4 dog")!]
