//
//  User+CoreDataProperties.swift
//  
//
//  Created by 강지원 on 2020/11/19.
//
//

import Foundation
import CoreData


extension User {

    @nonobjc public class func fetchRequest() -> NSFetchRequest<User> {
        return NSFetchRequest<User>(entityName: "User")
    }

    @NSManaged public var currentDogSerialNumber: Int64
    @NSManaged public var currentPlan: Int64
    @NSManaged public var currentPlanAchievementRate: Double
    @NSManaged public var graduateDogSerialList: [Int]?
    @NSManaged public var graduatePlanAchievementRates: [Double]?
    @NSManaged public var graduatePlanList: [Int]?
    @NSManaged public var pastJumpingJackRepsDict: [String: Int]?
    @NSManaged public var pastLungeRepsDict: [String: Int]?
    @NSManaged public var pastSquatRepsDict: [String: Int]?
    @NSManaged public var runAwayDogSerialList: [Int]?
    @NSManaged public var runAwayPlanAchievementRates: [Double]?
    @NSManaged public var runAwayPlanList: [Int]?
    @NSManaged public var todayAchievementRate: Double
    @NSManaged public var todayJumpingJackReps: Int64
    @NSManaged public var todayLungeReps: Int64
    @NSManaged public var todaySquatReps: Int64
    @NSManaged public var currentPlanDaysList: [String]?

}
