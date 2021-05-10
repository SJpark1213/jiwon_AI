//
//  StoreViewController.swift
//  Pages
//
//  Created by Leekyujin on 2020/11/19.
//

import StoreKit
import UIKit

class StoreViewController: UIViewController, UITableViewDelegate, UITableViewDataSource, SKProductsRequestDelegate, SKPaymentTransactionObserver {

    private var models = [SKProduct]()
    
    //MARK:- direction indicator
    let downArrow = UIImageView()

    private let tableView: UITableView = {
        let table = UITableView()
        table.register(UITableViewCell.self, forCellReuseIdentifier: "cell")
        
        
        
        return table
    }()
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        
        view.addSubview(tableView)
        SKPaymentQueue.default().add(self)
        tableView.delegate = self
        tableView.dataSource = self
        //        tableView.frame = view.bounds
        tableView.layer.cornerRadius = 25
        tableView.rowHeight = 100
        tableView.backgroundColor = .white
        tableView.translatesAutoresizingMaskIntoConstraints = false
        tableView.topAnchor.constraint(equalTo: view.centerYAnchor, constant: -250).isActive = true
        tableView.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 10).isActive = true
        tableView.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -10).isActive = true
        tableView.bottomAnchor.constraint(equalTo: view.centerYAnchor, constant: 150).isActive = true
        
        //MARK: - direction indicator

        view.addSubview(downArrow)
        downArrow.image = #imageLiteral(resourceName: "circle-arrow-down-512(white)")
        downArrow.alpha = 0.4
        downArrow.backgroundColor = .black
        downArrow.layer.cornerRadius = 75 / 2
        downArrow.clipsToBounds = true
        downArrow.translatesAutoresizingMaskIntoConstraints = false
        downArrow.centerXAnchor.constraint(equalTo: view.centerXAnchor).isActive = true
        downArrow.bottomAnchor.constraint(equalTo: view.centerYAnchor, constant: 310).isActive = true
        downArrow.widthAnchor.constraint(equalToConstant: 75).isActive = true
        downArrow.heightAnchor.constraint(equalToConstant: 75).isActive = true
    
        
        fetchProducts()
        
        // MARK: - Update
        additionalFetch()
        }
    
    // MARK: - Table
    
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return models.count
    }

    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let product = models[indexPath.row]
        //let cell = tableView.dequeueReusableCell(withIdentifier: "cell", for: indexPath)
        let cell: UITableViewCell = UITableViewCell(style: .subtitle, reuseIdentifier: "cell")


        
        //MARK:- cell additional textLabel
        let cellName = UILabel()
        cell.addSubview(cellName)
        cellName.text = "\(product.localizedTitle)"
        
        cellName.font = UIFont.boldSystemFont(ofSize: 22)
        cellName.translatesAutoresizingMaskIntoConstraints = false
        cellName.centerYAnchor.constraint(equalTo: cell.centerYAnchor, constant: -10).isActive = true
        cellName.leftAnchor.constraint(equalTo: cell.leftAnchor, constant: 100).isActive = true
        
        //MARK:- cell additional textLabel(subtitle)
        let cellsub = UILabel()
        cell.addSubview(cellsub)
        cellsub.text = "\(product.localizedDescription)"
        cellsub.textColor = .lightGray
        cellsub.font = UIFont.systemFont(ofSize: 13)
        cellsub.numberOfLines = 2
        cellsub.translatesAutoresizingMaskIntoConstraints = false
        cellsub.centerYAnchor.constraint(equalTo: cell.centerYAnchor, constant: 15).isActive = true
        cellsub.leftAnchor.constraint(equalTo: cell.leftAnchor, constant: 100).isActive = true
    
        
        // MARK:- cell imageView
        let cellImage = UIImageView()
        cell.addSubview(cellImage)
        cellImage.image = planListImage[indexPath.row]
        cellImage.clipsToBounds = true
        cellImage.layer.cornerRadius = 40
        cellImage.backgroundColor = .clear
        cellImage.contentMode = .scaleAspectFit
        cellImage.translatesAutoresizingMaskIntoConstraints = false
        cellImage.heightAnchor.constraint(equalToConstant: 80).isActive = true
        cellImage.widthAnchor.constraint(equalToConstant: 80).isActive = true
        cellImage.leftAnchor.constraint(equalTo: cell.leftAnchor, constant: 10).isActive = true
        cellImage.centerYAnchor.constraint(equalTo: cell.centerYAnchor).isActive = true
        
        
        //: \(product.localizedDescription)
        
        let button = UILabel()
        cell.addSubview(button)
        button.backgroundColor = #colorLiteral(red: 0.9371162057, green: 0.9303418398, blue: 0.9423027039, alpha: 1)
        button.text = "\(product.priceLocale.currencySymbol ?? "₩")\(product.price)"
        button.textAlignment = .center
        button.textColor = .systemBlue
        button.clipsToBounds = true
        button.layer.cornerRadius = 15
        button.translatesAutoresizingMaskIntoConstraints = false
        button.rightAnchor.constraint(equalTo: cell.rightAnchor, constant: -15).isActive = true
        button.centerYAnchor.constraint(equalTo: cell.centerYAnchor).isActive = true
        button.widthAnchor.constraint(equalToConstant: 80).isActive = true
        button.heightAnchor.constraint(equalToConstant: 35).isActive = true
        
        
        
        return cell
    }
    
        func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
            tableView.deselectRow(at: indexPath, animated: true)
            //Show purchase
            let payment = SKPayment(product: models[indexPath.row])
            SKPaymentQueue.default().add(payment)
        }
    
    
    
   
    
    
    //MARK: - Product
    
    enum product: String, CaseIterable {
        case planA = "com.dreamin.SwithDog.1week"
        case planB = "com.dreamin.SwithDog.2week"
        case planC = "com.dreamin.SwithDog.3week"
        case planD = "com.dreamin.SwithDog.4week"
    }
    
    private func fetchProducts() {
        let request = SKProductsRequest(productIdentifiers: Set(product.allCases.compactMap({ $0.rawValue })))
        request.delegate = self
        request.start()
    }
    
    func productsRequest(_ request: SKProductsRequest, didReceive response: SKProductsResponse) {
        DispatchQueue.main.async {
            print("Count: \(response.products.count)")
            self.models = response.products
            self.tableView.reloadData()
        }
    }
    
    func paymentQueue(_ queue: SKPaymentQueue, updatedTransactions transactions: [SKPaymentTransaction]) {
        // no impl.
        transactions.forEach({
            switch $0.transactionState{
            case .purchasing:
                print("purchasing")
                print("\(transactions[0].payment.productIdentifier)")
            case .purchased:
                print("purchased")
                let dateFormatter = DateFormatter()
                dateFormatter.dateFormat = "yyyy-MM-dd"
                
                let userDefaults = UserDefaults.standard
                nowUser.currentDogSerialNumber = userDefaults.integer(forKey: "totalDogNumbers") + 1
                userDefaults.setValue(userDefaults.integer(forKey: "totalDogNumbers") + 1, forKey: "totalDogNumbers")
                
                let planName = String(transactions[0].payment.productIdentifier).components(separatedBy: ".")[3]
                if planName == "1week"{
                    print("1week")
                    nowUser.currentPlan = 1
                    nowUser.currentPlanDaysList = []
                    nowUser.currentPlanDaysList.append(dateFormatter.string(from: Date()))

                    for i in 1...7{
                        let plusDay = Date(timeIntervalSinceNow: TimeInterval(86400 * i))
                        nowUser.currentPlanDaysList.append(dateFormatter.string(from: plusDay))
                    }
                    
                    nowUser.graduatePlanList.append(1)
                    print(nowUser.graduatePlanList)
                }else if planName == "2week"{
                    nowUser.currentPlan = 2
                    nowUser.currentPlanDaysList = []
                    nowUser.currentPlanDaysList.append(dateFormatter.string(from: Date()))

                    for i in 1...14{
                        let plusDay = Date(timeIntervalSinceNow: TimeInterval(86400 * i))
                        nowUser.currentPlanDaysList.append(dateFormatter.string(from: plusDay))
                    }
                    
                    nowUser.graduatePlanList.append(2)
                }else if planName == "3week"{
                    nowUser.currentPlan = 3
                    nowUser.currentPlanDaysList = []
                    nowUser.currentPlanDaysList.append(dateFormatter.string(from: Date()))

                    for i in 1...21{
                        let plusDay = Date(timeIntervalSinceNow: TimeInterval(86400 * i))
                        nowUser.currentPlanDaysList.append(dateFormatter.string(from: plusDay))
                    }
                    
                    nowUser.graduatePlanList.append(3)
                }else if planName == "4week"{
                    nowUser.currentPlan = 4
                    nowUser.currentPlanDaysList = []
                    nowUser.currentPlanDaysList.append(dateFormatter.string(from: Date()))

                    for i in 1...28{
                        let plusDay = Date(timeIntervalSinceNow: TimeInterval(86400 * i))
                        nowUser.currentPlanDaysList.append(dateFormatter.string(from: plusDay))
                    }
                    
                    nowUser.graduatePlanList.append(4)
                }
                
                SKPaymentQueue.default().finishTransaction($0)
            case .failed:
                
                SKPaymentQueue.default().finishTransaction($0)
            case .restored:
                break
            case .deferred:
                break
            @unknown default:
                break
            }
        })
    }
}


