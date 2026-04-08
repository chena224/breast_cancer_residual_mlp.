import matplotlib.pyplot as plt
import torch
import random
import numpy as np
import os
from torch.utils.data import Dataset,TensorDataset,DataLoader,random_split
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts



# 中文字体配置（保证可视化中文正常显示）
from matplotlib import font_manager
chinese_fonts = [f.name for f in font_manager.fontManager.ttflist if any(k in f.name.lower() for k in ['hei', 'song', 'yahei'])]
plt.rcParams["font.family"] = chinese_fonts[0] if chinese_fonts else "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

config={
    'experiment_name':'breast_cancer_residual_mlp',
    'seed':42,
    'device':'cuda' if torch.cuda.is_available() else 'cpu',

    'data':{
        'data_dir':'./data',
        'input_dim':30,
        'num_classes':2,
        'val_splite':0.1,
        'batch_size':32,
        'num_workers':0
    },

    'model':{
        'hidden_dims':[128,64,32],
        'activation':'ReLU',
        'use_residual':True,
        'dropout_rate':[0.2,0.2,0.1],
        'normalization':'batch'
    },

    'training':{
        'epochs':50,
        'optimizer':{
            'type':'AdamW',
            'lr':0.001,
            'weight_decay':1e-4
        },
        'scheduler':{
            'type':'CosineAnnealingWarmRestarts',
            'T_0':25,
            'eta_min':1e-6
            },
        'grad_clip':1.0,
        'early_stopping':{
            'patience':10,
            'mode':'max'
        }
    },
    
    'logging':{
        'enable_tensorboard':True,
        'log_dir':'./runs',
        'save_dir':'./models'
    }
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path,exist_ok=True)
        print(f"创建目录: {path}")

class TransformedSubset(Dataset):
    def __init__(self,subset,transform=None):
        self.subset=subset
        self.transform=transform
    def __getitem__(self, idx):
        x,y=self.subset[idx]
        if self.transform is not None:
            x=self.transform(x)
        return x.float(),y.long()
    def __len__(self):
        return len(self.subset)
    
def plot_training_history(history,save_path):
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(history['train_loss'],label='训练损失',linewidth=1.5)
    plt.plot(history['val_loss'],label='验证损失',linewidth=1.5)
    plt.xlabel('训练轮次(Epoch)')
    plt.ylabel('损失值(Loss)')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1,2,2)
    plt.plot(history['train_acc'],label='训练准确率',linewidth=1.5)
    plt.plot(history['val_acc'],label='验证准确率',linewidth=1.5)
    plt.xlabel('训练轮次(Epoch)')
    plt.ylabel('准确率(Accuracy)')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path,dpi=300,bbox_inches='tight')
    plt.show()
    plt.close()

def plot_confusion_matrix(all_targets,all_preds,num_classes,save_path):
    cm=confusion_matrix(all_targets,all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(
        cm,annot=True,fmt='d',cmap='Blues',
        xticklabels=['恶性','良性'] if num_classes==2 else [f'预测{i}' for i in range(num_classes)],
        yticklabels=['恶性','良性'] if num_classes==2 else [f'预测{i}' for i in range(len(num_classes))]
    )
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def analyze_errors(model,test_loader,device,num_classes,save_path):
    model.eval()
    all_preds=[]
    all_probs=[]
    all_targets=[]
    all_features=[]

    with torch.no_grad():
        for features,targets in test_loader:
            features,targets=features.to(device),targets.to(device)
            output=model(features)
            probs=torch.softmax(output,dim=1)

            all_preds.extend(torch.argmax(probs,dim=1).cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_features.extend(features.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    all_probs=np.concatenate(all_probs,axis=0)
    
    all_preds_np=np.array(all_preds)
    all_targets_np=np.array(all_targets)
    errors=all_preds_np!=all_targets_np
    error_indices=np.where(errors)[0]

    if len(error_indices)==0:
        print("模型在测试集上无错误！")
        return
    
    error_probs=all_probs[error_indices,all_preds_np[error_indices]]
    sorted_indices_by_prob=np.argsort(-error_probs)[:min(10,len(error_indices))]
    final_error_indices=error_indices[sorted_indices_by_prob]

    fig,axes=plt.subplots(2,5,figsize=(20,8)) if len(error_indices)>=10 else plt.subplots(1,len(final_error_indices),figsize=(5*len(final_error_indices),6))
    axes=axes.flatten() if len(final_error_indices)>=1 else [axes]
    feature_names = ['半径均值', '纹理均值', '周长均值', '面积均值', '光滑度均值'] 
    for i,idx in enumerate(final_error_indices):
        feat=all_features[idx][:5]
        true_label='恶性' if all_targets_np[idx]==0 else '良性'
        pred_label='恶性' if all_preds_np[idx]==0 else '良性'
        prob=all_probs[idx,all_preds_np[idx]]

        axes[i].bar(feature_names,feat,color=['red'if true_label!=pred_label else 'blue'])
        axes[i].set_title(f'真实:{true_label}\n预测:{pred_label}\n概率{prob:.2f}')
        axes[i].tick_params(axis='x',rotation=45)
        axes[i].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path,dpi=300,bbox_inches='tight')
    plt.show()
    plt.close()

def load_and_prepare_data(config):
        data_config=config['data']
        data=load_breast_cancer()
        X=data.data
        y=data.target

        scale=StandardScaler()
        X=scale.fit_transform(X)

        X_tensor=torch.tensor(X,dtype=torch.float32)
        y_tensor=torch.tensor(y,dtype=torch.long)
        full_dataset=TensorDataset(X_tensor,y_tensor)

        val_size=int(data_config['val_splite']*len(full_dataset))
        test_size=val_size
        train_size=len(full_dataset)-val_size-test_size

        train_dataset,val_dataset_raw,test_dataset=random_split(
            full_dataset,[train_size,val_size,test_size],
            generator=torch.Generator().manual_seed(config['seed'])
        )

        val_dataset=TransformedSubset(val_dataset_raw,transform=None)

        train_loader=DataLoader(train_dataset,batch_size=data_config['batch_size'],shuffle=True,
                                num_workers=data_config['num_workers'],pin_memory=True)

        val_loader = DataLoader(
            val_dataset, batch_size=data_config['batch_size'], shuffle=False,
            num_workers=data_config['num_workers'], pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=data_config['batch_size'], shuffle=False,
            num_workers=data_config['num_workers'], pin_memory=True
        )
        
        print(f"数据加载完成：训练集{len(train_dataset)}样本，验证集{len(val_dataset)}样本，测试集{len(test_dataset)}样本")
        print(f"特征维度：{data_config['input_dim']}，类别数：{data_config['num_classes']}")
        return train_loader, val_loader, test_loader
    
class ResidualBlock(nn.Module):
        def __init__(self,in_features,out_features,dropout_rate=0.0,normalization='batch',activation='ReLU'):
            super().__init__()
            self.activation=self._get_activation(activation)

            main_layers=[]
            if normalization:
                main_layers.append(self._get_normalization(normalization, in_features))
            main_layers.append(self.activation)
            main_layers.append(nn.Dropout(dropout_rate))
            main_layers.append(nn.Linear(in_features,out_features))

            if normalization:
                main_layers.append(self._get_normalization(normalization, out_features))
            main_layers.append(self.activation)
            main_layers.append(nn.Dropout(dropout_rate))
            main_layers.append(nn.Linear(out_features,out_features))

            self.main_path=nn.Sequential(*main_layers)
            self.shortcut=nn.Linear(in_features,out_features) if in_features!=out_features else nn.Identity()


        def _get_activation(self,act_name):
            if act_name=='ReLU':
                return nn.ReLU(inplace=True)
            elif act_name=='GELU':
                return nn.GELU()
            else:
                raise ValueError(f"不支持的激活函数：{act_name}")
            
        def _get_normalization(self,norm_type,num_features):
            if norm_type=='batch':
                return nn.BatchNorm1d(num_features)
            elif norm_type=='layer':
                return nn.LayerNorm(num_features)
            else:
                raise ValueError(f"不支持的归一化类型：{norm_type}")
            
        def forward(self,x):
            return self.main_path(x)+self.shortcut(x)
    
class AdvancedANN(nn.Module):
        def __init__(self,config):
            super().__init__()
            model_config=config['model']
            input_dim=config['data']['input_dim']
            num_classes=config['data']['num_classes']

            layers=[]
            layers.append(nn.Linear(input_dim,model_config['hidden_dims'][0]))

            for i in range(1,len(model_config['hidden_dims'])):
                if model_config['use_residual']:
                    layers.append(ResidualBlock(
                        in_features=model_config['hidden_dims'][i-1],
                        out_features=model_config['hidden_dims'][i],
                        dropout_rate=model_config['dropout_rate'][i-1],
                        normalization=model_config['normalization'],
                        activation=model_config['activation']
                    ))
                else:
                    if model_config['normalization']:
                        layers.append(self._get_normalization(model_config['normalization'],model_config['hidden_dims'][i-1]))
                    layers.append(self._get_activation(model_config['activation']))
                    layers.append(nn.Dropout(model_config['dropout_rate'][i-1]))
                    layers.append(nn.Linear(model_config['hidden_dims'][i-1],model_config['hidden_dims'][i]))
                
            if model_config['normalization']:
                layers.append(self._get_normalization(model_config['normalization'],model_config['hidden_dims'][-1]))
            layers.append(self._get_activation(model_config['activation']))
            layers.append(nn.Dropout(model_config['dropout_rate'][-1] if model_config['dropout_rate'] else 0.0))        
            layers.append(nn.Linear(model_config['hidden_dims'][-1],num_classes))

            self.model=nn.Sequential(*layers)
        def _get_activation(self,act_name):
            if act_name=='ReLU':
                return nn.ReLU(inplace=True)
            elif act_name=='GELU':
                return nn.GELU()
            else:
                raise ValueError(f"不支持的激活函数：{act_name}")
            
        def _get_normalization(self,norm_type,num_features):
            if norm_type=='batch':
                return nn.BatchNorm1d(num_features)
            elif norm_type=='layer':
                return nn.LayerNorm(num_features)
            else:
                raise ValueError(f"不支持的归一化类型：{norm_type}")
        
        def forward(self,x):
            return self.model(x)
        
def train_one_epoch(model,train_loader,criterion,optimizer,device,gard_clip=0):
        model.train()
        total_loss=0.0
        correct=0.0
        total=0.0

        loop=tqdm(train_loader,total=len(train_loader),leave=True,desc='训练')

        for features,targets in loop:
            features,targets=features.to(device),targets.to(device)

            optimizer.zero_grad()
            outputs=model(features)
            loss=criterion(outputs,targets)

            loss.backward()
            if gard_clip>0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),gard_clip)
            optimizer.step()

            total_loss+=loss.item()
            _,pred=torch.max(outputs.detach(),1)
            total+=targets.size(0)
            correct+=(pred==targets).sum().item()

            loop.set_postfix(
                loss=f'{total_loss/len(loop):.4f}',
                acc=f'{correct/total:.4f}'
            )
        
        avg_loss=total_loss/len(train_loader)
        accuracy=correct/total
        return avg_loss,accuracy
def validate(model,val_loader,criterion,device):
        model.eval()
        total_loss=0.0
        correct=0
        total=0
        all_targets=[]
        all_preds=[]

        with torch.no_grad():
            loop=tqdm(val_loader,total=len(val_loader),leave=True,desc='验证')
            for features,targets in loop:
                features,targets=features.to(device),targets.to(device)

                outputs=model(features)
                loss=criterion(outputs,targets)

                total_loss+=loss.item()
                _,preds=torch.max(outputs.detach(),1)
                total+=targets.size(0)
                correct+=(preds==targets).sum().item()

                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        avg_loss=total_loss/len(val_loader)
        accuracy=correct/total
        return avg_loss,accuracy,np.array(all_targets),np.array(all_preds)

def main():
    set_seed(config['seed'])
    device=torch.device(config['device'])
    print(f"使用设备：{device}")

    save_dir=os.path.join(config['logging']['save_dir'],config['experiment_name'])
    ensure_dir(save_dir)
    best_model_path=os.path.join(save_dir,'best_model.pth')

    train_loader,val_loader,test_loader=load_and_prepare_data(config)

    model=AdvancedANN(config).to(device)
    print("\n模型结构：")
    print(model)

    criterion=nn.CrossEntropyLoss()

    optimizer_config=config['training']['optimizer']
    if optimizer_config['type']=='AdamW':
        optimizer=optim.AdamW(
            model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config['weight_decay']

        )
    else:
        raise ValueError(f"不支持的优化器：{optimizer_config['type']}")
    
    scheduler_config=config['training']['scheduler']
    if scheduler_config['type']=='CosineAnnealingWarmRestarts':
        scheduler=CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_config['T_0'],
            eta_min=scheduler_config['eta_min']
        )
    else:
        scheduler=None

    history={
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []       
    }
    best_val_acc = 0.0
    patience_counter = 0
    
    print("\n" + "="*50)
    print("开始训练")
    print("="*50)
    for epoch in range(config['training']['epochs']):
        print(f"\n[轮次 {epoch+1}/{config['training']['epochs']}]")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config['training']['grad_clip']
        )
        
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        if scheduler is not None:
            scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_loss': train_loss,
                'val_loss': val_loss
            }, best_model_path)
            print(f"✅ 保存最佳模型！验证准确率：{best_val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"❌ 验证准确率未提升（当前最佳：{best_val_acc:.4f}），早停计数：{patience_counter}/{config['training']['early_stopping']['patience']}")
            
            if patience_counter >= config['training']['early_stopping']['patience']:
                print(f"\n早停触发！训练停止于轮次 {epoch+1}")
                break
    
    # 最终评估
    print("\n" + "="*50)
    print("开始最终测试（加载最佳模型）")
    print("="*50)
    
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"加载最佳模型（轮次 {checkpoint['epoch']}，验证准确率 {checkpoint['best_val_acc']:.4f}）")
    
    test_loss, test_acc, test_targets, test_preds = validate(model, test_loader, criterion, device)
    print(f"\n📊 测试集结果：损失={test_loss:.4f}，准确率={test_acc:.4f}")
    
    print("\n分类报告：")
    print(classification_report(
        test_targets, test_preds,
        target_names=['恶性', '良性'],  # 适配二分类
        digits=4
    ))
    
    # 可视化结果
    print("\n" + "="*50)
    print("生成可视化结果")
    print("="*50)
    
    train_history_path = os.path.join(save_dir, '训练历史曲线.png')
    plot_training_history(history, train_history_path)
    print(f"训练历史曲线已保存至：{train_history_path}")
    
    cm_path = os.path.join(save_dir, '混淆矩阵.png')
    plot_confusion_matrix(test_targets, test_preds, config['data']['num_classes'], cm_path)
    print(f"混淆矩阵已保存至：{cm_path}")
    
    # 错误分析：移除image_size参数（表格数据无需）
    error_analysis_path = os.path.join(save_dir, '错误案例分析.png')
    analyze_errors(model, test_loader, device, config['data']['num_classes'], error_analysis_path)
    print(f"错误案例分析已保存至：{error_analysis_path}")
    
    print("\n" + "="*50)
    print("训练与评估全部完成！")
    print(f"最佳模型路径：{best_model_path}")
    print(f"可视化结果目录：{save_dir}")
    print("="*50)

# ==============================================================================
# 运行主函数
# ==============================================================================
if __name__ == '__main__':
    main()



    
    
    


        



    

        


    

        














 

