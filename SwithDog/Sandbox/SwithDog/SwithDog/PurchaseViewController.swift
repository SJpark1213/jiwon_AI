//
//  PurchaseViewController.swift
//  SwithDog
//
//  Created by Woongbi Kim on 2020/10/23.
//  Copyright © 2020 Leekyujin. All rights reserved.
//
import StoreKit
import UIKit

class PurchaseViewController: UIViewController, SKProductsRequestDelegate, SKPaymentTransactionObserver {


    var myProduct: SKProduct?

    
    
    override func viewDidLoad() {
        super.viewDidLoad()

        // Do any additional setup after loading the view.
    fetchProducts()
    }
    func fetchProducts(){
        //com.example.dreamin.SwithDog
        let request = SKProductsRequest(productIdentifiers: ["com.example.dreamin.SwithDog"])
        request.delegate = self
        request.start()
    }
    
    @IBAction func didTapPurchase(){
        guard let myProduct = myProduct else {
            return
        }
        if SKPaymentQueue.canMakePayments() {
            let payment = SKPayment(product: myProduct)
            SKPaymentQueue.default().add(self)
            SKPaymentQueue.default().add(payment)
        }
    }

    func productsRequest(_ request: SKProductsRequest, didReceive response: SKProductsResponse) {
        if let product = response.products.first{
            myProduct = product
            print(product.productIdentifier)
            print(product.priceLocale)
            print(product.price)
            print(product.localizedTitle)
            print(product.localizedDescription)

        }
    }
    
    func paymentQueue(_ queue: SKPaymentQueue, updatedTransactions transactions: [SKPaymentTransaction]) {
        for transaction in transactions{
            switch transaction.transactionState {
            case .purchasing:
                //작동 없음
            break
            case .purchased, .restored:
                // 이미 구매하였음
                
                UserDefaults.standard.set(true, forKey: "UpgradePlans")
                
                SKPaymentQueue.default().finishTransaction(transaction)
                SKPaymentQueue.default().remove(self)
            break
            case .failed, .deferred:
                SKPaymentQueue.default().finishTransaction(transaction)
                SKPaymentQueue.default().remove(self)
            break
            default:
                SKPaymentQueue.default().finishTransaction(transaction)
                SKPaymentQueue.default().remove(self)
                break
            }
        }
        
    }
    
    
}
