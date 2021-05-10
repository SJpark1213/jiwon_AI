//
//  Model.swift
//  SwithDog
//
//  Created by Leekyujin on 2020/10/22.
//  Copyright © 2020 Leekyujin. All rights reserved.
//

import Foundation


class PlanPurchase {
    var content: String
    
    init(content: String) {
        self.content = content
    }
    
    static var  planList = [
        PlanPurchase(content: "1 day"),
        PlanPurchase(content: "3 days"),
        PlanPurchase(content: "1 week"),
        PlanPurchase(content: "2 weeks"),
        PlanPurchase(content: "1 month")
]
}
