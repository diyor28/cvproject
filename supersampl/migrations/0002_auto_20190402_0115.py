# Generated by Django 2.1.5 on 2019-04-01 20:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('supersampl', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='imagemodel',
            name='intermediate',
            field=models.ImageField(blank=True, upload_to=''),
        ),
        migrations.AddField(
            model_name='imagemodel',
            name='processed',
            field=models.ImageField(blank=True, upload_to=''),
        ),
    ]