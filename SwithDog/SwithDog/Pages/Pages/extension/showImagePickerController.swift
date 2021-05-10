//
//  showImagePickerController.swift
//  Pages
//
//  Created by Leekyujin on 2020/11/13.
//
import UIKit

extension ViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    func showImagePickerControllerActionSheet() {
        
        let imagePickerAlert = UIAlertController(title: "프로필 사진 선택", message: nil, preferredStyle: .actionSheet)
        let photoLibrayrAction = UIAlertAction(title: "앨범에서 가져오기", style: .default) { (action) in
            self.showimagePikcerController(sourceType: .photoLibrary)
        }
        let cameraAction = UIAlertAction(title: "사진 찍기", style: .default) { (action) in
            self.showimagePikcerController(sourceType: .camera)
        }
        let cancelAction = UIAlertAction(title: "취소", style: .cancel, handler: nil)
        
        imagePickerAlert.addAction(photoLibrayrAction)
        imagePickerAlert.addAction(cameraAction)
        imagePickerAlert.addAction(cancelAction)

        present(imagePickerAlert, animated: true, completion: nil)
    }
    
    func showimagePikcerController(sourceType: UIImagePickerController.SourceType) {
        let imagePicker = UIImagePickerController()
        imagePicker.delegate = self
        imagePicker.allowsEditing = true
        imagePicker.sourceType = sourceType
        
        present(imagePicker, animated: true, completion: nil)

    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        
        if let editedImage = info[UIImagePickerController.InfoKey.editedImage] as? UIImage {
            imageView.image = editedImage
        } else if let originalImage = info[UIImagePickerController.InfoKey.originalImage] as? UIImage {
            imageView.image = originalImage
        }
        
        dismiss(animated: true, completion: nil)
    }
}
