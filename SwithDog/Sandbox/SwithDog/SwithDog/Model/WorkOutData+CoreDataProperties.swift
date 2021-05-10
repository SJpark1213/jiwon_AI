//
//  WorkOutData+CoreDataProperties.swift
//  SwithDog
//
//  Created by Woongbi Kim on 2020/10/31.
//  Copyright © 2020 Leekyujin. All rights reserved.
//
//

import Foundation
import CoreData


extension WorkOutData {

    @nonobjc public class func fetchRequest() -> NSFetchRequest<WorkOutData> {
        return NSFetchRequest<WorkOutData>(entityName: "WorkOutData")
    }

    @NSManaged public var name: String?
    @NSManaged public var count: Int64

}

extension WorkOutData : Identifiable {

}
