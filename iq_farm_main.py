"""IQ-FARM"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from io import BytesIO

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, 
    MessageHandler, filters, ContextTypes, ConversationHandler
)
import config 
from arabic_reshaper import arabic_reshaper
from bidi.algorithm import get_display
import tempfile



# ============================================================================
# CONFIGURATION
# ============================================================================
TOKEN = config.BOT_TOKEN 
ADMIN_ID = config.ADMIN_USER_ID


# ============================================================================
# DATA MANAGER CLASS
# ============================================================================
class DataManager:
    """إدارة بيانات التربة ومتطلبات المحاصيل والتوصيات"""
    
    def __init__(self, soil_csv_path='datasets/soil_data.csv', crop_csv_path='datasets/crop_data.csv'):
        self.soil_csv_path = soil_csv_path
        self.crop_csv_path = crop_csv_path
        self.load_data()
    
    def load_data(self):
        """تحميل البيانات من ملفات CSV، إنشاء الملفات إذا لم تكن موجودة"""
        if not os.path.exists('datasets'):
            os.makedirs('datasets')
        
        # Load or create soil data
        if os.path.exists(self.soil_csv_path):
            self.soil_df = pd.read_csv(self.soil_csv_path)
        else:
            self._create_default_soil_data()
        
        # Load or create crop data
        if os.path.exists(self.crop_csv_path):
            self.crop_df = pd.read_csv(self.crop_csv_path)
        else:
            self._create_default_crop_data()
    
    def _create_default_soil_data(self):
        """إنشاء مجموعة البيانات الافتراضية للتربة العراقية"""
        soil_data = {
            'region': ['البصرة', 'البصرة', 'الناصرية', 'بغداد', 'كركوك', 'الموصل',
                      'ديالى', 'الأنبار', 'السليمانية', 'أربيل', 'الحلة', 'كربلاء'],
            'soil_type': ['طين غَريزي', 'تربة مالحة', 'طمي غَريزي', 'طمي كلسي',
                         'طين طمي', 'طمي رقيق', 'طين غَريزي', 'رمل طمي',
                         'طين', 'تربة رسوبية ثقيلة ', 'طمي غَريزي', 'طين كلسي'],
            'ph': [7.8, 8.2, 7.5, 7.9, 7.6, 7.4, 7.7, 7.3, 7.2, 7.3, 7.6, 8.0],
            'nitrogen_ppm': [45, 30, 52, 48, 55, 58, 50, 35, 62, 60, 46, 42],
            'phosphorus_ppm': [22, 15, 28, 25, 30, 32, 26, 18, 35, 33, 23, 20],
            'potassium_ppm': [250, 200, 280, 260, 290, 310, 270, 220, 340, 320, 240, 230],
            'moisture_content_percent': [35, 25, 32, 28, 30, 32, 30, 20, 38, 35, 28, 26],
            'organic_matter_percent': [2.1, 1.5, 2.8, 2.2, 3.1, 3.5, 2.5, 1.8, 4.2, 3.8, 2.3, 1.9],
            'temperature_celsius': [28, 28, 27, 26, 22, 20, 25, 32, 18, 19, 27, 28],
            'rainfall_mm_annual': [180, 180, 150, 120, 280, 320, 200, 80, 650, 480, 140, 110]
        }
        self.soil_df = pd.DataFrame(soil_data)
        self.soil_df.to_csv(self.soil_csv_path, index=False)
        print(f"✅ تم إنشاء بيانات التربة الافتراضية في: {self.soil_csv_path}")
    
    def _create_default_crop_data(self):
        """إنشاء مجموعة البيانات الافتراضية لمتطلبات المحاصيل"""
        crop_data = {
            'crop_name': ['نخيل التمر', 'قمح', 'شعير', 'التمر', 'حمضيات', 'طماطم',
                         'خيار', 'باذنجان', 'بصل', 'بطاطا', 'زيتون', 'لوز',
                         'رمان', 'أرز', 'شمّام', 'بطيخ', 'قرع'],
            'min_temperature': [25, 10, 5, 22, 15, 15, 15, 20, 10, 10, 15, 15, 15, 20, 18, 20, 18],
            'max_temperature': [40, 25, 25, 45, 30, 30, 35, 35, 25, 20, 35, 35, 35, 30, 35, 35, 35],
            'min_ph': [7.0, 6.5, 6.0, 7.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 7.0, 6.5, 6.5, 6.0, 6.0, 6.0, 6.0],
            'max_ph': [8.5, 7.5, 7.5, 8.5, 7.5, 7.0, 7.0, 7.5, 7.5, 7.5, 8.5, 8.0, 8.0, 7.5, 7.0, 7.0, 7.0],
            'min_nitrogen_ppm': [30, 40, 35, 25, 50, 60, 55, 50, 45, 60, 30, 35, 40, 70, 50, 45, 50],
            'min_phosphorus_ppm': [12, 15, 12, 10, 20, 25, 20, 18, 15, 25, 10, 12, 15, 30, 20, 18, 20],
            'min_potassium_ppm': [180, 150, 140, 170, 200, 250, 240, 220, 180, 300, 150, 180, 200, 280, 250, 230, 240],
            'min_moisture_percent': [15, 25, 20, 12, 40, 50, 50, 45, 30, 50, 30, 25, 35, 60, 40, 50, 40],
            'min_rainfall_mm': [50, 200, 200, 30, 600, 400, 350, 300, 350, 450, 400, 350, 400, 1500, 400, 350, 350]
        }
        self.crop_df = pd.DataFrame(crop_data)
        self.crop_df.to_csv(self.crop_csv_path, index=False)
        print(f"✅ تم إنشاء بيانات متطلبات المحاصيل الافتراضية في: {self.crop_csv_path}")
    
    def get_recommended_crops(self, soil_params):
        """
        خوارزمية توصية بسيطة بناءً على معاملات التربة
        تُرجع المحاصيل التي تتطابق مع ظروف التربة
        
        soil_params: قاموس يحتوي على: temperature, rainfall, ph, nitrogen_ppm, 
                     phosphorus_ppm, potassium_ppm, moisture_content_percent
        """
        recommendations = []
        
        for _, crop in self.crop_df.iterrows():
            score = 0
            reasons = []
            
            # Check temperature range (0-25 points)
            if soil_params.get('temperature', 20) >= crop['min_temperature'] and \
               soil_params.get('temperature', 20) <= crop['max_temperature']:
                score += 25
            else:
                temp = soil_params.get('temperature', 20)
                if crop['min_temperature'] <= temp <= crop['max_temperature']:
                    score += 20
            
            # Check pH range (0-20 points)
            if soil_params.get('ph', 7.5) >= crop['min_ph'] and \
               soil_params.get('ph', 7.5) <= crop['max_ph']:
                score += 20
                reasons.append("✓ حموضة التربة مناسبة")
            else:
                reasons.append("✗ حموضة التربة غير مثالية")
                score += 5
            
            # Check nitrogen (0-15 points)
            if soil_params.get('nitrogen_ppm', 50) >= crop['min_nitrogen_ppm'] * 0.8:
                score += 15
                reasons.append("✓ النيتروجين كافٍ")
            else:
                reasons.append("⚠ النيتروجين منخفض")
            
            # Check rainfall (0-20 points)
            if soil_params.get('rainfall_mm', 200) >= crop['min_rainfall_mm'] * 0.7:
                score += 20
                reasons.append("✓ الأمطار مناسبة")
            else:
                reasons.append("⚠ الأمطار منخفضة")
            
            # Check moisture (0-20 points)
            if soil_params.get('moisture_content_percent', 30) >= crop['min_moisture_percent'] * 0.7:
                score += 20
                reasons.append("✓ الرطوبة مناسبة")
            else:
                reasons.append("⚠ الرطوبة منخفضة")
            
            if score >= 40:  # Only recommend if score >= 40
                recommendations.append({
                    'crop': crop['crop_name'],
                    'score': score,
                    'reasons': reasons
                })
        
        # Sort by score (descending)
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:config.TOP_RECOMMENDATIONS]  # Return top 5
    
    def add_soil_data(self, new_data_dict):
        """إضافة بيانات تربة جديدة إلى مجموعة البيانات"""
        new_df = pd.DataFrame([new_data_dict])
        self.soil_df = pd.concat([self.soil_df, new_df], ignore_index=True)
        self.soil_df.to_csv(self.soil_csv_path, index=False)
        return True
    
    def get_regions(self):
        """الحصول على قائمة بالمناطق الفريدة"""
        return sorted(self.soil_df['region'].unique().tolist())
    
    def get_soil_by_region(self, region):
        """الحصول على متوسط معاملات التربة لمنطقة معينة"""
        region_data = self.soil_df[self.soil_df['region'] == region]
        if region_data.empty:
            return None
        
        return {
            'temperature': region_data['temperature_celsius'].mean(),
            'rainfall_mm': region_data['rainfall_mm_annual'].mean(),
            'ph': region_data['ph'].mean(),
            'nitrogen_ppm': region_data['nitrogen_ppm'].mean(),
            'phosphorus_ppm': region_data['phosphorus_ppm'].mean(),
            'potassium_ppm': region_data['potassium_ppm'].mean(),
            'moisture_content_percent': region_data['moisture_content_percent'].mean(),
            'organic_matter_percent': region_data['organic_matter_percent'].mean()
        }


# ============================================================================
# VISUALIZATION MANAGER
# ============================================================================


class VisualizationManager:
    """إدارة طريقة عرض البيانات"""
    @staticmethod
    def fix_arabic_text(text):
        """إصلاح نص RTL للعرض الصحيح في matplotlib (اختياري)."""
        try:
            reshaped_text = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped_text)
            return bidi_text
        except:
            return text


    @staticmethod
    def create_combined_charts(recommendations, soil_params):
        """إنشاء كلا الرسمين البيانيين في شكل واحد - جنباً إلى جنب"""
        from math import pi
        
        # Create figure with 2 subplots (1 row, 2 columns)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # ===== LEFT CHART: Recommendations Bar Chart =====
        crops = [VisualizationManager.fix_arabic_text(r['crop']) for r in recommendations]
        scores = [r['score'] for r in recommendations]
        colors = ['#2ecc71' if s >= 80 else '#f39c12' if s >= 60 else '#e74c3c' for s in scores]
        
        ax1.barh(crops, scores, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_xlabel(VisualizationManager.fix_arabic_text('نسبة التوصية (%)'), fontsize=12, fontweight='bold')
        ax1.set_title(VisualizationManager.fix_arabic_text('أفضل المحاصيل الموصى بها'), fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 100)
        
        for i, (crop, score) in enumerate(zip(crops, scores)):
            ax1.text(score + 2, i, f'{score}%', va='center', fontweight='bold')
        
        # ===== RIGHT CHART: Soil Analysis Radar =====
        ax2.remove()
        ax2 = fig.add_subplot(122, projection='polar')
        
        categories = ['حموضة', 'نيتروجين', 'فوسفور', 'بوتاسيوم', 'رطوبة']
        categories_fixed = [VisualizationManager.fix_arabic_text(c) for c in categories]
        
        values = [
            (soil_params.get('ph', 7.5) / 8) * 100,
            min((soil_params.get('nitrogen_ppm', 50) / 70) * 100, 100),
            min((soil_params.get('phosphorus_ppm', 25) / 40) * 100, 100),
            min((soil_params.get('potassium_ppm', 250) / 400) * 100, 100),
            min((soil_params.get('moisture_content_percent', 30) / 60) * 100, 100),
        ]
        values += values[:1]
        
        angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
        angles += angles[:1]
        
        ax2.plot(angles, values, 'o-', linewidth=2, color='#3498db')
        ax2.fill(angles, values, alpha=0.25, color='#3498db')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories_fixed, fontsize=10)
        ax2.set_ylim(0, 100)
        ax2.set_title(VisualizationManager.fix_arabic_text('تحليل جودة التربة'), fontsize=14, fontweight='bold', pad=20)
        ax2.grid(True)
        
        plt.tight_layout()
        return fig



    @staticmethod
    def create_recommendation_chart(recommendations):
        """إنشاء رسم بياني شريطي لتوصيات المحاصيل"""
        fig, ax = plt.subplots(figsize=(10, 6))
        crops = [VisualizationManager.fix_arabic_text(r['crop']) for r in recommendations]
        scores = [r['score'] for r in recommendations]
        colors = ['#2ecc71' if s >= 80 else '#f39c12' if s >= 60 else '#e74c3c' for s in scores]
        
        ax.barh(crops, scores, color=colors, edgecolor='black', linewidth=1.5)
        
        # Labels
        ax.set_xlabel(VisualizationManager.fix_arabic_text('نسبة التوصية (%)'), fontsize=12, fontweight='bold')
        ax.set_title(VisualizationManager.fix_arabic_text('أفضل المحاصيل الموصى بها'), fontsize=14, fontweight='bold')
        ax.set_xlim(0, 100)
        
        for i, (crop, score) in enumerate(zip(crops, scores)):
            ax.text(score + 2, i, f'{score}%', va='center', fontweight='bold')
        
        plt.tight_layout()
        return fig


    
    @staticmethod
    def create_soil_analysis_chart(soil_params):
        """إنشاء رسم بياني راداري لتحليل التربة مع نص عربي صحيح"""
        from math import pi
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Categories for radar chart
        categories = ['حموضة', 'نيتروجين', 'فوسفور', 'بوتاسيوم', 'رطوبة']
        categories_fixed = categories
        
        values = [
            (soil_params.get('ph', 7.5) / 8) * 100,
            min((soil_params.get('nitrogen_ppm', 50) / 70) * 100, 100),
            min((soil_params.get('phosphorus_ppm', 25) / 40) * 100, 100),
            min((soil_params.get('potassium_ppm', 250) / 400) * 100, 100),
            min((soil_params.get('moisture_content_percent', 30) / 60) * 100, 100),
        ]
        values += values[:1]
        
        angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color='#3498db')
        ax.fill(angles, values, alpha=0.25, color='#3498db')
        ax.set_xticks(angles[:-1])
        
        ax.set_xticklabels(categories_fixed, fontsize=10)


        ax.set_ylim(0, 100)


        ax.set_title('تحليل جودة التربة', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def save_chart_to_bytes(fig):
        """حفظ الرسم البياني إلى بايتات للإرسال عبر تيليجرام"""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf


# ============================================================================
# TELEGRAM BOT HANDLERS
# ============================================================================


# Global data manager
data_manager = DataManager('datasets/soil_data.csv', 'datasets/crop_data.csv')
viz_manager = VisualizationManager()


# Store user data temporarily
user_data_store = {}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالج أمر البدء"""
    user_id = update.effective_user.id
    user_data_store[user_id] = {}
    
    welcome_text = """
🌾 أهلاً بك في IQ-FARM 🌾

نظام التوصيات الذكية للمحاصيل للمزارعين العراقيين

المميزات:
✓ توصيات بناءً على بيانات التربة والطقس
✓ رسوم بيانية وتحليلات مرئية
✓ لوحة إدارة لإضافة وتعديل البيانات

اختر خياراً للبدء:
    """
    
    keyboard = [
        [
            InlineKeyboardButton("🌍 اختيار المنطقة", callback_data='select_region'),
            InlineKeyboardButton("📊 إدخال مخصص", callback_data='custom_input')
        ],
        [
            InlineKeyboardButton("📈 عرض نموذج", callback_data='view_stats'),
            InlineKeyboardButton("ℹ️ حول البرنامج", callback_data='about')
        ]
    ]
    
    if user_id == ADMIN_ID:
        keyboard.append([InlineKeyboardButton("🔐 لوحة الإدارة", callback_data='admin_panel')])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    if update.message:
        await update.message.reply_text(welcome_text, reply_markup=reply_markup)
    elif update.callback_query:
        await update.callback_query.delete_message()
        await update.callback_query.message.reply_text(welcome_text, reply_markup=reply_markup)
    


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالجة جميع نقرات الأزرار"""
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    
    if query.data == 'select_region':
        regions = data_manager.get_regions()
        keyboard = [[InlineKeyboardButton(region, callback_data=f'region_{region}')] for region in regions]
        keyboard.append([InlineKeyboardButton("← رجوع", callback_data='back_main')])
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text("اختر منطقتك:", reply_markup=reply_markup)
    
    elif query.data.startswith('region_'):
        region = query.data.split('region_', 1)[1]
        soil_params = data_manager.get_soil_by_region(region)
        
        if soil_params:
            user_data_store[user_id]['soil_params'] = soil_params
            user_data_store[user_id]['region'] = region
            await show_recommendations(query, user_id)
        else:
            await query.edit_message_text("عذراً، لم نجد بيانات لهذه المنطقة")
    
    elif query.data == 'custom_input':
        await query.edit_message_text(
            "🌡️ من فضلك أدخل درجة الحرارة بالدرجات المئوية:\n(مثلاً: 28)"
        )
        context.user_data['step'] = 'temperature'
    
    elif query.data == 'view_stats':
        await query.edit_message_text("⏳ جارٍ تحميل الإحصائيات...")
        # Create visualization
        sample_params = {
            'temperature': 27,
            'rainfall_mm': 200,
            'ph': 7.6,
            'nitrogen_ppm': 50,
            'phosphorus_ppm': 25,
            'potassium_ppm': 260,
            'moisture_content_percent': 30
        }
        await query.message.delete()
        recommendations = data_manager.get_recommended_crops(sample_params)
        fig = viz_manager.create_recommendation_chart(recommendations)
        chart_bytes = viz_manager.save_chart_to_bytes(fig)
        keyboard = [[InlineKeyboardButton("← رجوع", callback_data='back_main')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.reply_photo(
            photo=chart_bytes,
            caption="📊 أفضل المحاصيل (نموذج)",
            reply_markup=reply_markup
        )


    elif query.data == 'about':
        about_text = """
📖 حول نظام IQ-FARM

نظام IQ-FARM منصة ذكية لتقديم توصيات المحاصيل استناداً إلى بيانات التربة والطقس

🎯 الهدف:
دعم المزارعين في اختيار المحاصيل وزيادة إنتاجية المحصول

📚 مصادر البيانات:
عينات من مناطق عراقية متعددة مثل البصرة، الناصرية، بغداد، الموصل، ديالى، السليمانية، أربيل

🔬 التكنولوجيا المستخدمة:
Python، Pandas، NumPy، Matplotlib
        """
        keyboard = [[InlineKeyboardButton("← رجوع", callback_data='back_main')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(about_text, reply_markup=reply_markup)
    
    elif query.data == 'admin_panel':
        if user_id == ADMIN_ID:
            admin_text = """
🔐 لوحة الإدارة

اختر إجراءً:
            """
            keyboard = [
                [InlineKeyboardButton("➕ إضافة بيانات تربة", callback_data='add_soil_data')],
                [InlineKeyboardButton("📋 عرض جميع البيانات", callback_data='view_all_data')],
                [InlineKeyboardButton("📊 إحصائيات الاستخدام", callback_data='usage_stats')],
                [InlineKeyboardButton("← رجوع", callback_data='back_main')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(admin_text, reply_markup=reply_markup)
    
    elif query.data == 'add_soil_data':
        if user_id == ADMIN_ID:
            await query.edit_message_text(
                "أرسل ملف .csv يحتوي على بيانات التربة الجديدة لإضافتها."
            )
            context.user_data['step'] = 'add_soil_csv'
    
    elif query.data == 'back_main':
        await start(update, context)
    


async def show_recommendations(query, user_id):
    """عرض توصيات المحاصيل"""
    soil_params = user_data_store.get(user_id, {}).get('soil_params', {})
    region = user_data_store.get(user_id, {}).get('region', 'غير معروفة')
    
    recommendations = data_manager.get_recommended_crops(soil_params)
    
    if not recommendations:
        await query.edit_message_text(
            f"❌ لم نجد توصيات للمنطقة {region}\n\n"
            "جرب الإدخال المخصص أو اختر منطقة أخرى"
        )
        return
    
    # Create recommendation text
    rec_text = f"التوصيات لمنطقة: {region}\n\n"
    rec_text += "المحاصيل المقترحة:\n\n"
    for i, rec in enumerate(recommendations, 1):
        rec_text += f"{i}️⃣ {rec['crop']}\n"
        rec_text += f"تقييم التوصية: {rec['score']}%\n"
        for reason in rec['reasons']:
            rec_text += f"   {reason}\n"
        rec_text += "\n"
        if i == 5:
            break
    keyboard = [
        [InlineKeyboardButton("← رجوع", callback_data='back_main')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    fig = viz_manager.create_combined_charts(recommendations, soil_params)
    chart_bytes = viz_manager.save_chart_to_bytes(fig)
    await query.delete_message()
    await query.message.reply_photo(photo=chart_bytes, caption=rec_text,reply_markup=reply_markup)
    
async def handle_custom_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالجة إدخال بيانات التربة المخصصة"""
    user_id = update.effective_user.id
    
    if 'step' not in context.user_data:
        return
    
    step = context.user_data['step']
    
    try:
        if step == 'temperature':
            temp = float(update.message.text)
            user_data_store[user_id]['soil_params'] = {
                'temperature': temp,
                'rainfall_mm': 200,
                'ph': 7.5,
                'nitrogen_ppm': 50,
                'phosphorus_ppm': 25,
                'potassium_ppm': 260,
                'moisture_content_percent': 30
            }
            context.user_data['step'] = 'rainfall'
            await update.message.reply_text("💧 أدخل معدل الأمطار السنوي (مم):\n(مثلاً: 250)")

        elif step == 'rainfall':
            rainfall = float(update.message.text)
            user_data_store[user_id]['soil_params']['rainfall_mm'] = rainfall
            context.user_data['step'] = 'ph'
            await update.message.reply_text("🧪 أدخل حموضة التربة pH (مثلاً: 7.5)")

        
        elif step == 'ph':
            ph = float(update.message.text)
            user_data_store[user_id]['soil_params']['ph'] = ph
            
            # Get recommendations
            recommendations = data_manager.get_recommended_crops(user_data_store[user_id]['soil_params'])
            
            rec_text = "🌾 التوصيات بناءً على بيانات التربة:\n\n"
            for i, rec in enumerate(recommendations, 1):
                rec_text += f"{i}️⃣ {rec['crop']} ({rec['score']}%)\n"
            
            # Send recommendations
            await update.message.reply_text(rec_text)
            
            # Create and send chart
            fig = viz_manager.create_combined_charts(recommendations, user_data_store[user_id]['soil_params'])
            chart_bytes = viz_manager.save_chart_to_bytes(fig)
            await update.message.reply_photo(photo=chart_bytes)
            
            context.user_data.clear()
            await start(update, context)
    
    except ValueError:
        await update.message.reply_text("❌ من فضلك أدخل رقماً صحيحاً! حاول مرة أخرى.")


async def handle_admin_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالجة إدخال لوحة الإدارة"""
    user_id = update.effective_user.id
    
    if user_id != ADMIN_ID:
        await update.message.reply_text("❌ أنت لست مسؤولاً")
        return
    
    if 'step' not in context.user_data or context.user_data['step'] != 'add_soil_csv':
        return
    
    try:
        if not update.message.document:
            await update.message.reply_text("❌ من فضلك أرسل ملف .csv صحيح")
            return
        
        file = await update.message.document.get_file()
        tmp_path = tempfile.mkdtemp()
        full_path = await file.download_to_drive(custom_path=os.path.join(tmp_path, 'new_soil_data.csv'))
        print(full_path)
        print(full_path.name)
        if not full_path or not full_path.name.endswith('.csv'):
            await update.message.reply_text("❌ الملف يجب أن يكون بصيغة CSV")
            return
        
        new_data_df = pd.read_csv(full_path)
        required_columns = ['region', 'soil_type', 'ph', 'nitrogen_ppm',
                            'phosphorus_ppm', 'potassium_ppm', 'moisture_content_percent',
                            'organic_matter_percent', 'temperature_celsius', 'rainfall_mm_annual']
        
        if not all(col in new_data_df.columns for col in required_columns):
            await update.message.reply_text("❌ ملف CSV يفتقد بعض الأعمدة المطلوبة")
            return
        
        # Add each row
        for _, row in new_data_df.iterrows():
            data = row.to_dict()
            data_manager.add_soil_data(data)
        
        await update.message.reply_text(f"✅ تمت إضافة بيانات التربة بنجاح!\nعدد الصفوف المضافة: {len(new_data_df)}")
        context.user_data.clear()
        await start(update, context)
        
    except Exception as e:
        await update.message.reply_text(f"❌ حدث خطأ: {str(e)}")


# ============================================================================
# MAIN APPLICATION
# ============================================================================


def main():
    """تشغيل البوت"""
    print("🚀 جارٍ تشغيل نظام IQ-FARM...")
    
    app = Application.builder().token(TOKEN).build()
    
    # Add handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_custom_input))
    app.add_handler(MessageHandler(filters.Document.FileExtension("csv"), handle_admin_input))
    
    # Run
    app.run_polling()


if __name__ == '__main__':
    main()
