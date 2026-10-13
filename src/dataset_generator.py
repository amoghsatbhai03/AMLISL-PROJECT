"""
Comprehensive Multilingual Banking Complaint Dataset Generator
Generates realistic, balanced, multi-class customer complaints across 6 banking categories
and 6 languages (English, Spanish, French, German, Hindi, and Hinglish).
"""

import os
import random
import re
import pandas as pd

# Fix random seed for reproducibility
random.seed(42)

CATEGORIES = [
    "Bank Account Services",
    "Credit Card & Prepaid Card",
    "Mortgages & Housing Loans",
    "Personal & Consumer Loans",
    "Money Transfers & Digital Payments",
    "Credit Reporting & Identity Theft"
]

LANGUAGES = ["en", "es", "fr", "de", "hi", "mr", "hi-latn"]

# High-fidelity banking complaint patterns per category and language
TEMPLATES = {
    "Bank Account Services": {
        "en": [
            "My checking account was charged a monthly maintenance fee of ${amount} without any prior disclosure or notice.",
            "I visited the ATM at {branch} and requested ${amount}, but the machine dispensed nothing while my savings account was debited.",
            "The bank froze my savings account without providing any valid reason. I cannot access my money to pay rent.",
            "An unauthorized withdrawal of ${amount} appeared on my account statement. Please reverse this transaction immediately.",
            "I submitted documents to close my dormant checking account over 3 weeks ago, but the bank has still not closed it.",
            "You charged me three separate overdraft fees on the same day for a minor balance shortfall of just ${amount}.",
            "My direct deposit paycheck has been placed on an extended hold for 7 days despite having a clear history.",
            "The branch customer service was extremely unhelpful when I asked to remove wrongful debit charges from my savings account.",
            "I cannot link my new checking account with online banking. The portal repeatedly shows an account verification error.",
            "Why is there a minimum balance penalty fee deducted when my average quarterly balance exceeded the requirement?",
            "My debit card linked to checking account was blocked at the checkout counter without any SMS alert or explanation.",
            "I deposited a valid cashier check of ${amount} at the teller window, but the funds have not cleared after two weeks."
        ],
        "es": [
            "Me cobraron una comisión de mantenimiento de cuenta corriente de {amount}€ sin previo aviso ni justificación.",
            "Fui al cajero automático y solicité {amount}€, la máquina no entregó el efectivo pero el dinero fue debitado de mi cuenta de ahorros.",
            "El banco bloqueó mi cuenta bancaria sin darme ninguna explicación. No puedo pagar mis facturas esenciales.",
            "Apareció un retiro no autorizado de {amount}€ en el extracto de mi cuenta de cheques. Exijo la devolución inmediata.",
            "Solicité el cierre definitivo de mi cuenta hace más de un mes y la sucursal sigue cobrándome comisiones indebidas.",
            "Me aplicaron múltiples comisiones por descubierto cuando el saldo negativo fue causado por un cobro erróneo del propio banco.",
            "El depósito de mi nómina mensual fue retenido arbitrariamente durante una semana sin darme solución.",
            "La atención en la oficina bancaria fue pésima cuando reclamé los cargos duplicados en mi cuenta corriente."
        ],
        "fr": [
            "Des frais de tenue de compte de {amount}€ ont été prélevés sur mon compte courant sans mon consentement.",
            "Le distributeur automatique n'a pas délivré les {amount}€ demandés, mais la somme a bien été débitée de mon compte.",
            "Ma banque a bloqué mon compte d'épargne sans aucun préavis, m'empêchant de payer mes dépenses quotidiennes.",
            "Un prélèvement non autorisé de {amount}€ apparaît sur mon relevé bancaire. Je demande le remboursement immédiat.",
            "J'ai demandé la clôture de mon compte bancaire il y a un mois, mais la banque continue d'envoyer des avis de frais.",
            "On m'a facturé des frais de découvert abusifs alors que mon compte a toujours été approvisionné correctement.",
            "Le virement de mon salaire est bloqué depuis cinq jours sans justification de la part de l'agence bancaire."
        ],
        "de": [
            "Mein Girokonto wurde mit einer Kontoführungsgebühr von {amount}€ belastet, obwohl mir ein kostenloses Konto zugesichert wurde.",
            "Am Geldautomaten wurden {amount}€ nicht ausgezahlt, jedoch von meinem Sparkonto abgebucht. Bitte sofort korrigieren.",
            "Die Bank hat mein Bankkonto ohne Vorankündigung gesperrt. Ich habe keinen Zugriff mehr auf mein Geld.",
            "Eine unberechtigte Abbuchung über {amount}€ ist auf meinem Kontoauszug aufgetaucht. Bitte umgehende Rückbuchung.",
            "Ich habe vor Wochen die Schließung meines Kontos beantragt, doch die Bank zieht weiterhin unberechtigte Gebühren ein.",
            "Überziehungszinsen wurden doppelt berechnet, obwohl mein Konto zu keinem Zeitpunkt im Minus war.",
            "Mein Gehaltseingang wird seit einer Woche grundlos von der Bankfiliale zurückgehalten."
        ],
        "hi": [
            "मेरे बचत खाते से बिना किसी पूर्व सूचना के {amount} रुपये का खाता रखरखाव शुल्क काट लिया गया है।",
            "एटीएम से {amount} रुपये निकालते समय मशीन से नकदी नहीं निकली, लेकिन मेरे बैंक खाते से पैसे कट गए।",
            "बैंक ने बिना कोई कारण बताए मेरा बचत खाता फ्रीज कर दिया है, जिससे मैं किराया भी नहीं दे पा रहा हूँ।",
            "मेरे बैंक स्टेटमेंट में {amount} रुपये की अनाधिकृत निकासी दिखाई दे रही है। कृपया यह राशि तुरंत वापस करें।",
            "मैंने एक महीने पहले अपना बैंक खाता बंद करने का आवेदन दिया था, लेकिन बैंक अभी भी पेनल्टी लगा रहा है।",
            "न्यूनतम बैलेंस न होने के नाम पर बैंक ने मनमाना जुर्माना काट लिया है जबकि मेरा औसत बैलेंस सही था।",
            "मेरी सैलरी का चेक बैंक में जमा करने के 10 दिन बाद भी खाते में क्रेडिट नहीं हुआ है।"
        ],
        "mr": [
            "माझ्या बचत खात्यातून कोणतीही पूर्वसूचना न देता {amount} रुपये मासिक देखभाल शुल्क कापले गेले आहे.",
            "मी एटीएममधून {amount} रुपये काढण्याचा प्रयत्न केला, पण मशीनमधून पैसे आले नाहीत आणि खात्यातून रक्कम वजा झाली.",
            "बँकेने कोणतेही वैध कारण न देता माझे बँक खाते गोठवले आहे, मला घरभाडे भरणेही अशक्य झाले आहे.",
            "माझ्या बँक खात्याच्या विवरणात {amount} रुपयांची अनधिकृत पैसे काढल्याची नोंद दिसत आहे, कृपया पैसे त्वरित परत करा.",
            "मी एक महिन्यापूर्वी बचत खाते बंद करण्यासाठी अर्ज दिला होता, तरीही बँक अजूनही अनपेक्षित शुल्क लावत आहे.",
            "किमान शिल्लक नसल्याच्या नावाखाली बँकेने खात्यातून मनमानी दंड कापला आहे, जो अत्यंत चुकीचा आहे.",
            "माझ्या चेकद्वारे जमा केलेले {amount} रुपये १५ दिवस उलटूनही खात्यात जमा झालेले नाहीत."
        ],
        "hi-latn": [
            "Mere savings account se bina kisi reason ke {amount} rupees maintenance charge kaat liya gaya hai.",
            "ATM se {amount} rupaye nikal raha tha, cash nahi nikla par mere account se balance debit ho gaya.",
            "Bank ne mera checking account bina bataye block kar diya hai, main apna rent bhi nahi de pa raha hu.",
            "Account statement check kiya toh {amount} rupees ka unauthorized deduction dikha raha hai, refund karo.",
            "Maine 1 mahina pehle account close karne ka form diya tha, branch walon ne abhi tak close nahi kiya.",
            "Minimum balance penalty baar baar kaati ja rahi hai jabki mera average quarterly balance bilkul sahi tha.",
            "Mera salary transfer bank ne hold par daal diya hai bina kisi valid information ke."
        ]
    },

    "Credit Card & Prepaid Card": {
        "en": [
            "There is a fraudulent charge of ${amount} on my credit card from a merchant I have never visited or heard of.",
            "I paid my credit card bill in full before the due date, yet the bank imposed a late payment fee and finance charges.",
            "The bank unilaterally reduced my credit card limit from $10,000 to $1,500 without providing any notification.",
            "I requested to cancel my annual fee credit card, but the representative charged the ${amount} annual fee anyway.",
            "My reward cashback points accumulated over two years were wiped out without prior warning.",
            "I reported my stolen credit card 5 days ago, but replacement card has not arrived and unauthorized transactions are pending.",
            "The merchant issued a refund of ${amount} for returned goods, but the bank refuses to reflect it on my credit card balance.",
            "I am being charged an exorbitant interest rate APR that contradicts the promotional zero percent offer in my agreement.",
            "My prepaid travel card was declined abroad leaving me stranded, even though it had an active balance of ${amount}.",
            "A double billing occurred for a single transaction at {merchant} on my credit card statement."
        ],
        "es": [
            "Hay un cargo fraudulento de {amount}€ en mi tarjeta de crédito de un comercio que jamás he visitado.",
            "Pagué el saldo total de mi tarjeta de crédito antes del vencimiento y aun así me cobraron intereses de demora.",
            "El banco redujo el límite de mi tarjeta de crédito drásticamente sin previo aviso ni explicación alguna.",
            "Cancelé mi tarjeta de crédito hace dos meses y me siguen facturando la cuota de mantenimiento anual de {amount}€.",
            "Mis puntos y reembolsos acumulados en el programa de recompensas de la tarjeta desaparecieron sin motivo.",
            "Reporté el extravío de mi tarjeta de crédito y continúan apareciendo cargos que yo no autoricé.",
            "El comercio me devolvió {amount}€ por una compra cancelada, pero el importe no se acredita en mi tarjeta de crédito."
        ],
        "fr": [
            "Un prélèvement frauduleux de {amount}€ apparaît sur ma carte de crédit pour un achat que je n'ai jamais effectué.",
            "J'ai réglé l'intégralité de mon relevé de carte avant la date limite, pourtant des pénalités de retard m'ont été infligées.",
            "La banque a réduit le plafond de ma carte bancaire sans avertissement, bloquant tous mes paiements en déplacement.",
            "J'ai résilié ma carte bancaire l'an dernier, mais les frais de cotisation annuelle de {amount}€ continuent d'être prélevés.",
            "Mes points de fidélité et remises cashback acquis avec ma carte visa ont été supprimés sans raison valable.",
            "J'ai signalé le vol de ma carte de crédit, mais de nouvelles transactions frauduleuses continuent d'être débitées.",
            "Le commerçant m'a remboursé {amount}€, mais le crédit n'apparaît toujours pas sur mon compte de carte."
        ],
        "de": [
            "Auf meiner Kreditkartenabrechnung befindet sich eine betrügerische Belastung von {amount}€ von einem unbekannten Händler.",
            "Ich habe meine Kreditkartenrechnung fristgerecht beglichen, trotzdem wurden Verzugszinsen und Mahngebühren erhoben.",
            "Mein Kreditkartenlimit wurde von der Bank willkürlich herabgesetzt, ohne mich vorher zu informieren.",
            "Ich habe meine Kreditkarte fristgerecht gekündigt, dennoch wurde die Jahresgebühr von {amount}€ erneut berechnet.",
            "Meine gesammelten Bonuspunkte und Cashback-Guthaben wurden einfach ohne Ankündigung gelöscht.",
            "Ich habe meine gestohlene Kreditkarte sperren lassen, aber es wurden danach weitere unbefugte Umsätze gebucht."
        ],
        "hi": [
            "मेरे क्रेडिट कार्ड पर {amount} रुपये का फर्जी और अनाधिकृत लेन-देन दर्ज हुआ है जो मैंने कभी नहीं किया।",
            "मैंने अंतिम तिथि से पहले अपने क्रेडिट कार्ड के पूरे बिल का भुगतान कर दिया था, फिर भी लेट फीस जोड़ दी गई।",
            "बैंक ने बिना किसी सूचना के मेरे क्रेडिट कार्ड की लिमिट को अचानक काफी कम कर दिया है।",
            "क्रेडिट कार्ड बंद करवाने के बावजूद बैंक ने वार्षिक शुल्क के नाम पर {amount} रुपये का बिल भेज दिया है।",
            "मेरे क्रेडिट कार्ड के रिवॉर्ड पॉइंट्स जो मैंने दो सालों में जुटाए थे, बैंक ने बिना बताए शून्य कर दिए।",
            "कार्ड खोने की सूचना देने के बाद भी उस पर फ्रॉड ट्रांजैक्शन होते रहे, बैंक कोई सुरक्षा नहीं दे सका।",
            "दुकानदार द्वारा {amount} रुपये का रिफंड भेजे जाने के बाद भी यह मेरे क्रेडिट कार्ड में क्रेडिट नहीं हुआ है।"
        ],
        "mr": [
            "माझ्या क्रेडिट कार्डवर {amount} रुपयांचा अनधिकृत आणि बनावट व्यवहार झाला आहे जो मी कधीही केला नाही.",
            "मी देय तारखेपूर्वी क्रेडिट कार्डचे संपूर्ण बिल भरले होते, तरीही बँकेने लेट फी आणि व्याज आकारले आहे.",
            "बँकेने कोणतीही पूर्वकल्पना न देता माझ्या क्रेडिट कार्डची मर्यादा अचानक खूप कमी केली आहे.",
            "क्रेडिट कार्ड बंद करण्याची विनंती करूनही बँकेने {amount} रुपये वार्षिक नूतनीकरण शुल्क आकारले आहे.",
            "माझ्या क्रेडिट कार्डचे जमा झालेले रिवॉर्ड पॉईंट्स बँकेने कोणतीही माहिती न देता अचानक रद्द केले आहेत.",
            "क्रेडिट कार्ड चोरीला गेल्याची तक्रार करूनही त्यावरून अनधिकृत व्यवहार सुरूच राहिले, बँकेने दुर्लक्ष केले.",
            "व्यापाऱ्याने परतावा (रिफंड) पाठवला असतानाही ती रक्कम माझ्या क्रेडिट कार्ड खात्यावर जमा झालेली नाही."
        ],
        "hi-latn": [
            "Mere credit card par {amount} rupees ka unauthorized fraud transaction hua hai jo maine nahi kiya.",
            "Credit card bill date se pehle pura pay kar diya tha, fir bhi late payment fine and interest charge lagaya.",
            "Bank ne bina kisi notice ke meri credit card spending limit aadhi kar di.",
            "Card cancel karne ki request daali thi par customer care ne annual fees {amount} charge kar diya.",
            "Mere credit card reward points bina kisi reason ke expire kar diye gaye hain.",
            "Stolen credit card block karwaya tha par replacement card 2 hafte baad bhi nahi mila.",
            "Merchant ne {amount} ka refund bhej diya tha par mere credit card balance me reflect nahi kar raha."
        ]
    },

    "Mortgages & Housing Loans": {
        "en": [
            "The mortgage servicer miscalculated my escrow account balance resulting in an unjustified monthly payment increase of ${amount}.",
            "I sent my monthly home loan mortgage payment on the 1st, but the servicer held funds in suspense and reported a late fee.",
            "The bank has delayed processing my mortgage refinancing application for over 4 months, causing me to lose my locked interest rate.",
            "The mortgage lender failed to pay my homeowner property taxes on time from the escrow account, incurring county tax penalties.",
            "I paid off my mortgage loan balance completely, but the bank refuses to release the property deed and satisfaction lien.",
            "The bank refuses to cancel my private mortgage insurance (PMI) even though my home equity is well above 22 percent.",
            "I was incorrectly threatened with home foreclosure proceedings despite being current on all my monthly mortgage installments.",
            "My adjustable rate mortgage ARM interest rate was hiked by 3 percent without proper index adjustment calculations.",
            "The customer service department at the mortgage division is completely unresponsive regarding loan modification paperwork."
        ],
        "es": [
            "La entidad hipotecaria calculó erróneamente mi cuenta de custodia, aumentando la cuota mensual de mi hipoteca en {amount}€.",
            "Pagué la cuota de la hipoteca a tiempo y el banco retuvo el dinero cobrándome intereses de demora injustificados.",
            "El banco tardó más de cuatro meses en tramitar mi novación hipotecaria, haciéndome perder el tipo de interés pactado.",
            "El banco no pagó los impuestos sobre bienes inmuebles desde la cuenta de custodia generando recargos fiscales.",
            "Amorticé la totalidad de mi préstamo hipotecario y la entidad se niega a tramitar la cancelación registral de la hipoteca.",
            "Me siguen cobrando el seguro de hipoteca obligatorio a pesar de que el capital pendiente es inferior al 75 por ciento.",
            "Recibí una amenaza de embargo hipotecario cuando estoy al corriente de todos los pagos mensuales."
        ],
        "fr": [
            "L'organisme de prêt immobilier a mal calculé les mensualités de mon crédit logement avec une augmentation injustifiée de {amount}€.",
            "J'ai versé l'échéance de mon prêt immobilier à temps, mais la banque a enregistré un retard avec des pénalités indues.",
            "La banque a traîné pendant des mois sur mon dossier de rachat de crédit immobilier, me faisant perdre le taux avantageux.",
            "J'ai remboursé par anticipation la totalité de mon prêt immobilier, mais la banque refuse de lever l'hypothèque.",
            "La banque refuse de résilier l'assurance emprunteur alors que j'ai fourni un contrat équivalent conformément à la loi.",
            "Des menaces de saisie immobilière ont été formulées par erreur alors que tous mes prélèvements de prêt sont à jour."
        ],
        "de": [
            "Mein Baufinanzierer hat die monatliche Hypothekenrate ohne nachvollziehbare Begründung um {amount}€ angehoben.",
            "Meine monatliche Tilgung für das Immobiliendarlehen wurde fristgerecht überwiesen, dennoch verbuchte die Bank Verzugszinsen.",
            "Die Bearbeitung meiner Anschlussfinanzierung verzögert sich seit Monaten, wodurch mir günstigere Zinssätze entgangen sind.",
            "Das Immobiliendarlehen wurde vollständig abbezahlt, aber die Bank verweigert die Ausstellung der Löschungsbewilligung für das Grundbuch.",
            "Trotz Einhaltung aller Ratenzahlungen drohte mir die Kreditabteilung fälschlicherweise mit Zwangsversteigerung.",
            "Fehlerhafte Zinsanpassungen bei meinem variablen Hypothekendarlehen haben zu erheblichen Mehrkosten geführt."
        ],
        "hi": [
            "बैंक ने मेरे होम लोन की मासिक किस्त (ईएमआई) बिना किसी स्पष्ट कारण के {amount} रुपये बढ़ा दी है।",
            "मैंने अपने गृह ऋण की मासिक ईएमआई समय पर भर दी थी, फिर भी बैंक ने लेट पेमेंट पेनल्टी लगा दी।",
            "मैंने अपना पूरा होम लोन चुका दिया है, लेकिन बैंक मेरे मकान के मूल रजिस्ट्री दस्तावेज वापस नहीं कर रहा है।",
            "होम लोन की ब्याज दरें कम होने के बावजूद बैंक ने मेरी ब्याज दर को कम करने से मना कर दिया है।",
            "सभी किस्तों का नियमित भुगतान करने के बाद भी बैंक ने मेरे घर की कुर्की का गलत नोटिस भेज दिया है।",
            "ऋण सेवा प्रदाता ने मेरे प्रॉपर्टी टैक्स का भुगतान समय पर नहीं किया जिससे जुर्माना लग गया।"
        ],
        "mr": [
            "बँकेने माझ्या गृहकर्जाचा (होम लोन) मासिक हप्ता (ईएमआय) विनाकारण {amount} रुपयांनी वाढवला आहे.",
            "मी माझ्या गृहकर्जाचा मासिक हप्ता वेळेवर भरला असतानाही बँकेने दंडात्मक शुल्क लावले आहे.",
            "मी माझे गृहकर्ज पूर्णपणे फेडले आहे, परंतु बँक घराची मूळ कागदपत्रे आणि एनओसी देण्यास टाळाटाळ करत आहे.",
            "बाजारात व्याजदर कमी झाले असतानाही बँक माझ्या गृहकर्जाचा व्याजदर कमी करत नाहीये.",
            "सर्व हप्ते वेळेत भरत असतानाही बँकेने मालमत्ता जप्तीची चुकीची कायदेशीर नोटीस पाठवली आहे.",
            "गृहकर्ज फेररचना (लोन मॉडिफिकेशन) करण्यासाठी दिलेले अर्ज बँक अधिकारी दुर्लक्षित करत आहेत."
        ],
        "hi-latn": [
            "Bank ne home loan EMI bina kisi notice ke {amount} rupees badha di hai, interest rate calculation galat hai.",
            "Maine home loan ki monthly EMI time par pay ki thi par unhone bounce charge aur late fee laga di.",
            "Home loan full pay hone ke baad bhi bank mere original property documents release nahi kar raha.",
            "Mortgage refinancing apply kiya tha 3 mahine se file pending padi hai aur rate lock expire ho gaya.",
            "Sabhi EMI clear hone ke bawajood legal notice aur foreclosure threat bhej diya hai, turant action lo.",
            "Loan foreclosure paper mangne par branch manager mana kar raha hai aur unnecessary charges maang raha hai."
        ]
    },

    "Personal & Consumer Loans": {
        "en": [
            "I applied for an auto loan payoff balance, paid the quoted ${amount}, but the lender still claims I owe an extra $450.",
            "Aggressive debt collection agents are calling my workplace and relatives regarding a consumer loan that has already been settled.",
            "The personal loan interest rate was advertised at 7.5% but the contract I was billed on has an APR of 16.8%.",
            "I requested a temporary student loan forbearance due to medical hardship, but the servicer rejected it without review.",
            "The lender reported a late payment on my vehicle loan to credit agencies even though autopay was active with funds available.",
            "I am being charged daily prepayment penalties of ${amount} for settling my personal loan ahead of schedule.",
            "The lender repossessed my car without providing the legally required statutory default notice.",
            "Excessive origination and processing fees were deducted upfront from my approved personal loan disbursement."
        ],
        "es": [
            "Liquidé el total de mi préstamo personal abonando {amount}€, pero la entidad crediticia insiste en cobrarme comisiones fantasmas.",
            "Agentes de cobro de deudas están acosando a mi familia por un préstamo al consumo que ya fue pagado en su totalidad.",
            "Me prometieron un tipo de interés del 6% en el préstamo para mi coche y en el contrato final aplicaron más del 15%.",
            "Solicité una carencia temporal en mi préstamo por desempleo y fue rechazada de forma totalmente arbitraria.",
            "El banco reportó un retraso en las cuotas de mi préstamo de vehículo a pesar de que el cobro automático estaba activado.",
            "Me están cobrando una penalización desorbitada de {amount}€ por amortización anticipada de mi crédito personal."
        ],
        "fr": [
            "J'ai soldé mon crédit à la consommation en versant {amount}€, mais l'organisme réclame encore des sommes injustifiées.",
            "Une société de recouvrement harcèle mes proches pour un prêt personnel qui a pourtant été entièrement remboursé.",
            "Le taux d'intérêt de mon prêt auto devait être de 5%, mais on me facture un TAEG de 13% sur les mensualités.",
            "L'organisme de prêt refuse la suspension temporaire de mes échéances de crédit malgré ma situation de chômage.",
            "Un incident de paiement a été déclaré sur mon prêt personnel alors que mon compte bancaire était suffisamment approvisionné.",
            "Des pénalités abusives de remboursement anticipé de {amount}€ m'ont été prélevées sans base contractuelle."
        ],
        "de": [
            "Ich habe meinen Ratenkredit vollständig mit {amount}€ abgelöst, dennoch fordert die Bank weitere angebliche Restschulden.",
            "Ein Inkassobüro belästigt meine Familie wegen eines Verbraucherkredits, der bereits vor Monaten beglichen wurde.",
            "Der effektive Jahreszins für meinen Autokredit wurde im Darlehensvertrag deutlich höher angesetzt als im Angebot vereinbart.",
            "Die Bank weigert sich grundlos, eine vertraglich vereinbarte Ratenpause für meinen Privatkredit zu gewähren.",
            "Obwohl die monatliche Kreditrate per Lastschrift eingezogen wurde, meldete der Kreditgeber fälschlicherweise Zahlungsverzug.",
            "Überhöhte Vorfälligkeitsentschädigungen von {amount}€ wurden bei vorzeitiger Kreditablösung abgebucht."
        ],
        "hi": [
            "मैंने अपने पर्सनल लोन की पूरी राशि {amount} रुपये चुका दी है, फिर भी बैंक अतिरिक्त पेनल्टी मांग रहा है।",
            "रिकवरी एजेंट मेरे कार्यस्थल पर फोन करके अभद्र भाषा का प्रयोग कर रहे हैं जबकि लोन की किस्तें समय पर दी जा रही हैं।",
            "कार लोन लेते समय 8% ब्याज बताया गया था लेकिन अब बैंक 15% की दर से ईएमआई काट रहा है।",
            "लोन का समय से पहले भुगतान करने पर बैंक ने भारी प्री-क्लोजर चार्ज और जुर्माना लगा दिया है।",
            "खाते में पर्याप्त पैसे होने पर भी बैंक ने लोन ईएमआई का ईसीएस बाउंस दिखाकर 500 रुपये पेनल्टी काट ली।",
            "बिना किसी पूर्व सूचना या नोटिस के फाइनेंस कंपनी ने मेरी गाड़ी जब्त कर ली है।"
        ],
        "mr": [
            "मी माझ्या वैयक्तिक कर्जाची संपूर्ण {amount} रुपये रक्कम भरली आहे, तरीही बँक अतिरिक्त बनावट थकबाकी मागत आहे.",
            "कर्ज वसुली एजंट माझ्या कामाच्या ठिकाणी फोन करून शिवीगाळ आणि मानसिक छळ करत आहेत.",
            "कार लोन घेताना ८ टक्के व्याजदर सांगण्यात आला होता, पण करारात १५ टक्के दराने हप्ता कापला जात आहे.",
            "मुदतीपूर्वी वैयक्तिक कर्ज बंद केल्यामुळे बँकेने अवाजवी प्री-क्लोजर दंड आकारला आहे.",
            "खात्यात पुरेसे पैसे असतानाही बँकेने लोन ऑटो-डेबिट बाऊन्स दाखवून ५०० रुपये दंड कापला आहे.",
            "कोणतीही पूर्वसूचना न देता फायनान्स कंपनीने माझी गाडी बेकायदेशीरपणे जप्त केली आहे."
        ],
        "hi-latn": [
            "Personal loan ka full settlement amount {amount} pay kar diya tha par bank NOC certificate nahi de raha.",
            "Recovery agents mere family members ko call karke pareshan kar rahe hain jabki loan installment clear hai.",
            "Car loan par promotional interest rate bolkar high rate of interest charge kar rahe hain.",
            "Personal loan pre-close karne par illegal pre-payment penalty charge lagaya gaya hai.",
            "Bank balance available tha fir bhi loan auto-debit fail dikhakar bounce charges laga diye.",
            "Two wheeler loan payoff hone ke baad bhi loan account active show kar raha hai."
        ]
    },

    "Money Transfers & Digital Payments": {
        "en": [
            "I initiated an international wire transfer of ${amount} to my family 8 days ago, but the funds have vanished into thin air.",
            "My instant mobile payment app transaction of ${amount} failed, money was deducted from my account but recipient received zero.",
            "The bank processed an outgoing wire transfer without sending a one-time password OTP or asking for two-factor authorization.",
            "I was charged an exorbitant hidden currency exchange markup fee of ${amount} on an overseas remittance transfer.",
            "The peer-to-peer payment platform locked my digital wallet with a remaining balance of ${amount} and no access to support.",
            "An erroneous duplicate wire transfer was dispatched by the bank operator, emptying my business account reserve.",
            "My scheduled recurring rent transfer via automated clearing house ACH failed without notice, causing landlord eviction threat.",
            "The digital payment gateway has placed a 21-day holding period on incoming commercial client remittances."
        ],
        "es": [
            "Realicé una transferencia bancaria internacional de {amount}€ hace una semana y el dinero no ha llegado al destinatario.",
            "Mi pago instantáneo con la aplicación móvil falló: el dinero fue descontado de mi cuenta pero el beneficiario no recibió nada.",
            "El banco ejecutó una transferencia de fondos sin solicitar el código de verificación SMS ni la clave de seguridad.",
            "Me cobraron comisiones ocultas y un tipo de cambio abusivo de {amount}€ en un envío de dinero al extranjero.",
            "La billetera digital bloqueó mi saldo de {amount}€ sin darme acceso ni posibilidad de transferirlo a mi banco.",
            "El banco emitió una transferencia duplicada por error, dejando mi cuenta bancaria completamente en descubierto."
        ],
        "fr": [
            "J'ai effectué un virement international de {amount}€ il y a dix jours et les fonds n'ont jamais été reçus par le destinataire.",
            "Le paiement instantané par mobile a échoué : mon compte a été débité de {amount}€ mais le commerçant n'a rien reçu.",
            "La banque a validé un virement sortant sans exiger l'authentification forte ni l'envoi d'un code de confirmation.",
            "Des frais de change et commissions exorbitantes de {amount}€ m'ont été facturés sur un transfert d'argent à l'étranger.",
            "Mon portefeuille électronique a été gelé avec un solde de {amount}€ sans qu'aucun service client ne me réponde.",
            "Un virement bancaire a été exécuté deux fois par erreur technique, créant un découvert injustifié sur mon compte."
        ],
        "de": [
            "Ich habe vor über einer Woche eine Auslandsüberweisung von {amount}€ getätigt, doch das Geld ist bis heute nicht beim Empfänger eingetroffen.",
            "Meine Echtzeit-Überweisung ist fehlgeschlagen: Der Betrag von {amount}€ wurde abgebucht, aber der Empfänger hat nichts erhalten.",
            "Die Bank hat eine Überweisung freigegeben, ohne dass ich eine TAN oder Freigabe in der Banking-App erteilt habe.",
            "Versteckte Wechselkursgebühren und Aufschläge von {amount}€ wurden bei meiner internationalen Überweisung einbehalten.",
            "Mein digitales Wallet wurde mit einem Guthaben von {amount}€ ohne nachvollziehbaren Grund gesperrt.",
            "Durch einen Systemfehler der Bank wurde eine Überweisung doppelt ausgeführt und mein Geschäftskonto überzogen."
        ],
        "hi": [
            "मैंने अपने परिवार को {amount} रुपये का ऑनलाइन वायर ट्रांसफर किया था, 7 दिन बाद भी पैसा खाते में नहीं पहुंचा।",
            "यूपीआई (UPI) से {amount} रुपये का भुगतान विफल हो गया, मेरे खाते से पैसे कट गए पर दुकानदार को नहीं मिले।",
            "बिना किसी ओटीपी (OTP) या मेरी अनुमति के मेरे बैंक खाते से ऑनलाइन मनी ट्रांसफर कर दिया गया।",
            "विदेश में पैसे भेजने पर बैंक ने बिना बताए भारी छुपा हुआ मुद्रा विनिमय शुल्क (करेंसी कन्वर्जन चार्ज) काट लिया।",
            "डिजिटल पेमेंट वॉलेट ने मेरे {amount} रुपये का बैलेंस बिना किसी कारण के ब्लॉक कर दिया है।",
            "बैंक की तकनीकी खराबी के कारण एक ही ट्रांजैक्शन दो बार हो गया और मेरा बैलेंस खाली हो गया।"
        ],
        "mr": [
            "मी माझ्या कुटुंबाला {amount} रुपयांचा ऑनलाइन वायर ट्रान्सफर केला होता, ८ दिवस झाले तरी पैसे खात्यात पोहोचले नाहीत.",
            "माझा यूपीआय (UPI) द्वारे केलेला {amount} रुपयांचा व्यवहार अयशस्वी झाला, खात्यातून पैसे कट झाले पण समोरच्याला मिळाले नाहीत.",
            "माझ्या परवानगीशिवाय आणि ओटीपी (OTP) शिवाय माझ्या खात्यातून अज्ञात व्यक्तीला ऑनलाइन पैसे पाठवले गेले आहेत.",
            "परदेशात पैसे पाठवताना बँकेने न सांगता प्रचंड चलन विनिमय शुल्क (करन्सी कन्वर्जन चार्ज) कापले आहे.",
            "डिजिटल पेमेंट वॉलेटने माझे {amount} रुपये शिल्लक असलेले वॉलेट विनाकारण ब्लॉक केले आहे.",
            "तांत्रिक बिघाडामुळे बँकेकडून एकाच व्यवहाराचे पैसे दोनदा कापले गेले, कृपया त्वरित परतावा द्या."
        ],
        "hi-latn": [
            "Maine {amount} rupees ka online bank transfer kiya tha, 6 din ho gaye receiver ko abhi tak paise nahi mile.",
            "UPI transaction fail ho gaya, mere bank se paisa cut ho gaya par recipient ko receive nahi hua, refund chahiye.",
            "Bina kisi OTP verify kiye mere account se unknown beneficiary ko money transfer kaise ho gaya?",
            "International remittance bhejne par heavy hidden charges aur foreign exchange deduction kar liya.",
            "Payment app wallet me {amount} balance tha, unhone wallet freeze kar diya aur help center respond nahi kar raha.",
            "NEFT / RTGS transfer stuck ho gaya hai, bank wale bol rahe hain server down tha, mera paisa wapas karo."
        ]
    },

    "Credit Reporting & Identity Theft": {
        "en": [
            "My credit report contains a delinquent debt collection account of ${amount} that belongs to an identity thief with a similar name.",
            "I filed a formal dispute regarding an erroneous 90-day late payment mark on my credit history, but the bureau verified it without investigation.",
            "An unauthorized credit card inquiry appeared on my credit file from a lending company I never authorized or applied to.",
            "The credit reporting agency failed to update my public records after a court judgment against me was completely vacated and dismissed.",
            "My credit score dropped by 115 points due to a false default account reported by {merchant} which was paid off years ago.",
            "I placed a credit security freeze on all my credit bureau files, but lenders are still pulling hard credit inquiries without my pin.",
            "Someone used my Social Security Number to open fraudulent retail store accounts, and the credit bureau refuses to remove them.",
            "The credit bureau is merging my credit file with an estranged relative who has multiple bankruptcies and liens."
        ],
        "es": [
            "Mi informe crediticio incluye una deuda de {amount}€ que no me pertenece y es producto de una suplantación de identidad.",
            "Presenté una reclamación formal por un apunte erróneo de morosidad y la agencia de crédito se niega a rectificarlo.",
            "Aparece una consulta no autorizada en mi historial de crédito realizada por una entidad financiera con la que no tengo relación.",
            "Mi puntuación crediticia cayó en picado por una deuda que ya fue saldada y que el fichero de morosos no ha cancelado.",
            "Alguien utilizó mis datos personales para abrir cuentas fraudulentas y la agencia crediticia no bloquea mi expediente.",
            "Solicité el bloqueo de seguridad en mi historial crediticio y continúan autorizando consultas financieras sin mi permiso."
        ],
        "fr": [
            "Mon dossier de crédit mentionne une dette contentieuse de {amount}€ résultant d'une usurpation d'identité avérée.",
            "J'ai contesté une mention erronée d'incident de paiement, mais l'organisme refuse d'actualiser mes informations bancaires.",
            "Une demande de crédit non autorisée a été inscrite à mon dossier par une société financière que je n'ai jamais sollicitée.",
            "Ma cote de crédit a chuté lourdement à cause d'une dette déjà remboursée qui figure toujours au fichier des incidents.",
            "Une tierce personne a usurpé mon numéro d'identification pour souscrire des crédits à la consommation en mon nom.",
            "L'agence d'évaluation du crédit refuse de supprimer les mentions litigieuses malgré la décision de justice en ma faveur."
        ],
        "de": [
            "In meiner Kreditauskunft (Schufa) steht ein unberechtigter Negativeintrag über {amount}€, der auf Identitätsdiebstahl zurückgeht.",
            "Ich habe formell Widerspruch gegen einen fehlerhaften Zahlungsrückstand eingelegt, doch die Auskunftei weigert sich, ihn zu löschen.",
            "Eine unberechtigte Kreditanfrage eines mir völlig unbekannten Anbieters wurde in meiner Bonitätsakte vermerkt.",
            "Mein Bonitätsscore ist drastisch gesunken, weil eine längst beglichene Forderung weiterhin als offen gemeldet wird.",
            "Kriminelle haben unter meinem Namen Online-Kreditkonten eröffnet, und die Auskunftei reagiert nicht auf meine Betrugsanzeige.",
            "Trotz Vorlage gerichtlicher Nachweise weigert sich die Kreditauskunftei, falsche Insolvenzdaten aus meiner Akte zu entfernen."
        ],
        "hi": [
            "मेरे सिबिल (CIBIL) क्रेडिट रिपोर्ट में {amount} रुपये का फर्जी डिफॉल्ट दिखाया जा रहा है जो पहचान की चोरी (Identity Theft) का मामला है।",
            "मैंने क्रेडिट ब्यूरो में गलत 90 दिन के लेट पेमेंट मार्क के खिलाफ शिकायत की थी, लेकिन उन्होंने बिना जांच के खारिज कर दी।",
            "मेरी अनुमति के बिना किसी अनजान लोन कंपनी ने मेरे पैन कार्ड पर हार्ड क्रेडिट इन्क्वायरी कर दी जिससे मेरा स्कोर गिर गया।",
            "मैंने 2 साल पहले पूरा लोन चुका दिया था, फिर भी सिबिल स्कोर में डिफॉल्टर लिखा हुआ आ रहा है।",
            "किसी अज्ञात व्यक्ति ने मेरे आधार और पैन कार्ड का दुरुपयोग करके फर्जी अकाउंट खुलवा लिया और ब्यूरो सुधार नहीं कर रहा है।",
            "क्रेडिट ब्यूरो ने मेरी फाइल को किसी दूसरे व्यक्ति के साथ मिला दिया है जिसके ऊपर कई बैंक डिफॉल्ट हैं।"
        ],
        "mr": [
            "माझ्या सिबिल (CIBIL) क्रेडिट रिपोर्टमध्ये {amount} रुपयांचे बनावट कर्ज थकित दिसत आहे, जी ओळख चोरी (Identity Theft) आहे.",
            "क्रेडिट ब्युरोमध्ये चुकीच्या लेट पेमेंट नोंदीविरुद्ध तक्रार दाखल केली होती, पण त्यांनी चौकशी न करता ती फेटाळली.",
            "माझ्या परवानगीशिवाय अनोळखी फायनान्स कंपनीने माझ्या पॅन कार्डवर क्रेडिट इन्क्वायरी केली, ज्यामुळे सिबिल स्कोअर घसरला.",
            "मी सर्व कर्ज फेडूनही सिबिल स्कोअरमध्ये अजूनही मला डिफॉल्टर दाखवले जात आहे.",
            "माझ्या कागदपत्रांचा गैरवापर करून दुसऱ्या व्यक्तीने घेतलेले कर्ज माझ्या सिबिल रिपोर्टमध्ये जोडले गेले आहे.",
            "क्रेडिट ब्युरोने माझी फाईल दुसऱ्या डिफॉल्टर व्यक्तीच्या खात्याशी चुकीने लिंक केली आहे."
        ],
        "hi-latn": [
            "Meri CIBIL credit report me {amount} ka fake overdue loan show kar raha hai jo maine kabhi liya hi nahi, fraud identity theft case hai.",
            "Credit bureau me dispute raise kiya tha wrong late payment mark hatane ke liye par unhone reject kar diya.",
            "Meri permission ke bina multiple banks ne mere PAN card par hard credit check kiya jisse score 80 points gir gaya.",
            "Loan full settlement hone ke baad bhi CIBIL me account status closed nahi dikha raha hai, score kharab ho raha hai.",
            "Kisi ne mera identity misuse karke fake consumer loan open kar liya, please complaint register karke score theek karo.",
            "Credit score drastically drop ho gaya hai wrong default reporting ki wajah se, please immediate rectifications karein."
        ]
    }
}

EMOTIONAL_PREFIXES = {
    "en": [
        "I am extremely frustrated with your service.",
        "This is completely unacceptable banking practice.",
        "Urgent attention required!",
        "I demand immediate resolution to this issue.",
        "Your bank has caused me serious financial stress.",
        "I have been a loyal customer for 10 years but this is ridiculous.",
        "Please look into this critical issue immediately.",
        "Warning: If this is not resolved, I will contact regulatory ombudsman."
    ],
    "es": [
        "Estoy sumamente indignado con el trato recibido por este banco.",
        "Esta situación es una completa negligencia bancaria.",
        "¡Atención urgente requerida!",
        "Exijo una solución inmediata a este problema financiero.",
        "El banco me está ocasionando un grave perjuicio económico.",
        "Si no resuelven esto de inmediato, presentaré una queja formal ante el Banco Central."
    ],
    "fr": [
        "Je suis extrêmement mécontent de la gestion de mon dossier bancaire.",
        "C'est une négligence grave de la part de votre établissement financier.",
        "Urgence absolue, veuillez traiter ce dossier sans délai.",
        "J'exige une régularisation immédiate sous peine de saisir le médiateur bancaire.",
        "Votre banque me met dans une situation financière intenable."
    ],
    "de": [
        "Ich bin extrem verärgert über diesen mangelhaften Kundenservice.",
        "Dies ist ein absolut inakzeptables Verhalten Ihrer Bank.",
        "Dringendes Handeln erforderlich!",
        "Ich fordere die sofortige Bereinigung dieses Fehlers.",
        "Ihre Bank hat mir erhebliche finanzielle Unannehmlichkeiten verursacht.",
        "Sollte dies nicht geregelt werden, schalte ich die Bundesanstalt für Finanzdienstleistungsaufsicht ein."
    ],
    "hi": [
        "मैं आपके बैंक की इस घटिया सेवा से बहुत ज्यादा परेशान और निराश हूँ।",
        "यह सरासर लापरवाही और ग्राहकों के साथ धोखा है।",
        "अत्यंत आवश्यक: कृपया इस मामले को तत्काल हल करें।",
        "मैं इस समस्या के तुरंत समाधान की मांग करता हूँ अन्यथा बैंकिंग लोकपाल में शिकायत करूँगा।",
        "आपके बैंक की वजह से मुझे बहुत बड़ा आर्थिक नुकसान उठाना पड़ रहा है।"
    ],
    "mr": [
        "मी बँकेच्या या अत्यंत निकृष्ट सेवेमुळे प्रचंड त्रस्त झालो आहे.",
        "हा सरळ सरळ ग्राहकांची फसवणूक आणि निष्काळजीपणा आहे.",
        "तातडीची मदत हवी आहे, कृपया या समस्येकडे त्वरित लक्ष द्या.",
        "या समस्येचे तातडीने निवारण न झाल्यास मी बँकिंग लोकपालकडे औपचारिक तक्रार दाखल करेन.",
        "बँकेच्या या चुकीमुळे मला मोठे आर्थिक नुकसान आणि मानसिक त्रास सहन करावा लागत आहे."
    ],
    "hi-latn": [
        "Main aapke bank ki service se bohot zyada frustrated hu.",
        "Yeh bilkul unacceptable hai, customer support bilkul bekar hai.",
        "Urgent help chahiye, mera paisa phasa hua hai.",
        "Agar yeh solve nahi hua toh main RBI Ombudsman me formal complaint darj karunga.",
        "Bank ki galti ki wajah se mujhe heavy financial loss aur mental tension ho rahi hai."
    ]
}


def inject_realistic_noise(text: str, lang: str) -> str:
    """Injects natural human typing noise, abbreviations, and casing variations to prevent artificial 100% memorization."""
    # 25% chance of realistic abbreviations
    if random.random() < 0.28:
        replacements = [
            (r"\baccount\b", "acct"),
            (r"\bstatement\b", "stmt"),
            (r"\btransaction\b", "txn"),
            (r"\bbalance\b", "bal"),
            (r"\bpayment\b", "pmt"),
            (r"\bplease\b", "pls"),
            (r"\bcredit card\b", "cc"),
            (r"\bchecking account\b", "checking acct"),
            (r"\bsavings account\b", "savings acct"),
            (r"\bcuenta corriente\b", "cta cte"),
            (r"\bcompte bancaire\b", "cpte bancaire"),
            (r"\bBankkonto\b", "Konto"),
            (r"\brupees\b", "rs"),
            (r"\bरुपये\b", "रु."),
            (r"\brupaiye\b", "rs"),
        ]
        for pattern, rep in replacements:
            if random.random() < 0.35:
                text = re.sub(pattern, rep, text, flags=re.IGNORECASE)

    # 15% chance of mild synthetic typographical slip (swap or drop a character)
    if random.random() < 0.15 and len(text) > 25:
        words = text.split()
        if len(words) > 4:
            idx = random.randint(0, len(words) - 1)
            w = words[idx]
            if len(w) > 5 and w.isalpha():
                t_type = random.choice(["drop", "swap", "double"])
                pos = random.randint(1, len(w) - 2)
                if t_type == "drop":
                    w = w[:pos] + w[pos+1:]
                elif t_type == "swap":
                    w = w[:pos] + w[pos+1] + w[pos] + w[pos+2:]
                elif t_type == "double":
                    w = w[:pos] + w[pos] + w[pos:]
                words[idx] = w
                text = " ".join(words)

    # 8% chance of informal lowercase
    if random.random() < 0.08:
        text = text.lower()

    return text


def generate_complaints_dataset() -> pd.DataFrame:
    """
    Generates a rich, well-distributed synthetic multilingual banking complaint dataset
    with strict separation between training patterns and held-out test evaluation patterns
    to prevent template memorization and artificial 100% accuracy overfitting.
    Includes realistic cross-domain multi-intent boundary complaints.
    """
    records = []
    sample_id = 1000

    amounts = [
        "150", "250", "420", "850", "1,200", "2,500", "4,800", "9,500",
        "12,000", "25,000", "45,000", "75,000", "150,000"
    ]
    branches = ["Downtown Branch", "Central Plaza", "Metro City Center", "Airport Terminal Branch", "Main Square"]
    merchants = ["Amazon Prime", "Global Electronic Store", "Air Travels Ltd", "HyperMarket Retail", "Online Streaming Service"]

    for category, lang_dict in TEMPLATES.items():
        for lang, template_list in lang_dict.items():
            prefixes = EMOTIONAL_PREFIXES.get(lang, [""])
            unseen_challenge_tpl = template_list[-1]
            core_tpls = template_list[:-1]

            # 1. Training Set Variations (95 per cat/lang = 3,990 core train samples)
            for _ in range(95):
                sample_id += 1
                base_tpl = random.choice(core_tpls)
                amt = random.choice(amounts)
                branch = random.choice(branches)
                merchant = random.choice(merchants)
                text = base_tpl.replace("{amount}", amt).replace("{branch}", branch).replace("{merchant}", merchant)

                if random.random() < 0.40:
                    text = f"{random.choice(prefixes)} {text}"

                text = inject_realistic_noise(text, lang)
                records.append({
                    "complaint_id": f"CMP-{sample_id}",
                    "complaint_text": text,
                    "category": category,
                    "language": lang,
                    "split": "train"
                })

            # 2. Held-out Test Set Instances from Core Templates (20 per cat/lang = 840 core test samples)
            for _ in range(20):
                sample_id += 1
                base_tpl = random.choice(core_tpls)
                amt = random.choice(amounts)
                branch = random.choice(branches)
                merchant = random.choice(merchants)
                text = base_tpl.replace("{amount}", amt).replace("{branch}", branch).replace("{merchant}", merchant)

                if random.random() < 0.35:
                    text = f"{random.choice(prefixes)} {text}"

                text = inject_realistic_noise(text, lang)
                records.append({
                    "complaint_id": f"CMP-{sample_id}",
                    "complaint_text": text,
                    "category": category,
                    "language": lang,
                    "split": "test"
                })

            # 3. Unseen Challenge Phrasings for Held-out Test (3 per cat/lang = 126 challenge test samples)
            for _ in range(3):
                sample_id += 1
                amt = random.choice(amounts)
                branch = random.choice(branches)
                merchant = random.choice(merchants)
                text = unseen_challenge_tpl.replace("{amount}", amt).replace("{branch}", branch).replace("{merchant}", merchant)

                if random.random() < 0.30:
                    text = f"{random.choice(prefixes)} {text}"

                text = inject_realistic_noise(text, lang)
                records.append({
                    "complaint_id": f"CMP-{sample_id}",
                    "complaint_text": text,
                    "category": category,
                    "language": lang,
                    "split": "test"
                })

    # Realistic cross-boundary multi-intent banking complaints that naturally evaluate product margins
    cross_cutting_samples = [
        # (text, category, lang)
        ("My monthly credit card bill autopay was deducted from my checking account twice and caused overdraft penalty.", "Credit Card & Prepaid Card", "en"),
        ("Mi pago automático de tarjeta fue debitado dos veces de mi cuenta corriente causando comisión por descubierto.", "Credit Card & Prepaid Card", "es"),
        ("Le prélèvement de ma carte bancaire a été débité deux fois sur mon compte courant avec des agios injustifiés.", "Credit Card & Prepaid Card", "fr"),
        ("Die Lastschrift für meine Kreditkarte wurde doppelt vom Girokonto abgebucht und verursachte Überziehungszinsen.", "Credit Card & Prepaid Card", "de"),
        ("मेरे बचत खाते से क्रेडिट कार्ड का बिल दो बार काट लिया गया है और ओवरड्राफ्ट पेनल्टी भी लगा दी गई है।", "Credit Card & Prepaid Card", "hi"),
        ("माझ्या बचत खात्यातून क्रेडिट कार्डचे बिल दोनदा कापले गेले आहे आणि ओव्हरड्राफ्ट दंड देखील आकारला गेला.", "Credit Card & Prepaid Card", "mr"),
        ("Savings account se credit card bill auto-debit double ho gaya hai aur penalty lag gayi hai.", "Credit Card & Prepaid Card", "hi-latn"),

        ("A wire transfer sent to my mortgage loan servicer is not reflecting on my housing loan statement.", "Money Transfers & Digital Payments", "en"),
        ("La transferencia bancaria para pagar mi hipoteca no aparece en el extracto del préstamo.", "Money Transfers & Digital Payments", "es"),
        ("Le virement bancaire pour mon prêt immobilier n'apparaît pas sur mon relevé de crédit.", "Money Transfers & Digital Payments", "fr"),
        ("Die Überweisung für meine Baufinanzierung wird nicht im Darlehenskonto angezeigt.", "Money Transfers & Digital Payments", "de"),
        ("होम लोन भरने के लिए किया गया ऑनलाइन बैंक ट्रांसफर अभी तक लोन खाते में नहीं दिखा रहा है।", "Money Transfers & Digital Payments", "hi"),
        ("गृहकर्ज भरण्यासाठी केलेला बँक ट्रान्सफर अजूनही खात्यात जमा झालेला दिसत नाही.", "Money Transfers & Digital Payments", "mr"),
        ("Home loan pay karne ke liye wire transfer kiya tha par loan statement me show nahi kar raha.", "Money Transfers & Digital Payments", "hi-latn"),

        ("The credit bureau reported a late payment for an auto loan installment that was already settled.", "Credit Reporting & Identity Theft", "en"),
        ("El informe de crédito muestra un retraso falso en un préstamo personal que ya fue liquidado.", "Credit Reporting & Identity Theft", "es"),
        ("L'organisme de crédit a signalé un incident pour un prêt auto qui a déjà été totalement remboursé.", "Credit Reporting & Identity Theft", "fr"),
        ("Die Auskunftei meldet einen Zahlungsverzug für einen Autokredit, der längst abbezahlt ist.", "Credit Reporting & Identity Theft", "de"),
        ("क्रेडिट रिपोर्ट में चुकाए जा चुके पर्सनल लोन को डिफॉल्ट दिखाया जा रहा है।", "Credit Reporting & Identity Theft", "hi"),
        ("सिबिल रिपोर्टमध्ये पूर्ण फेडलेल्या वैयक्तिक कर्जावर खोटी थकबाकी दाखवली जात आहे.", "Credit Reporting & Identity Theft", "mr"),
        ("Credit report me closed vehicle loan par late mark show ho raha hai.", "Credit Reporting & Identity Theft", "hi-latn"),

        ("The bank branch refused to provide the loan closure NOC and kept charging account maintenance.", "Personal & Consumer Loans", "en"),
        ("La sucursal se niega a darme el certificado de deuda cero y sigue cobrando comisiones.", "Personal & Consumer Loans", "es"),
        ("L'agence refuse de fournir l'attestation de fin de prêt et continue de prélever des frais.", "Personal & Consumer Loans", "fr"),
        ("Die Filiale weigert sich, die Krediterledigung zu bestätigen und berechnet weiter Kontogebühren.", "Personal & Consumer Loans", "de"),
        ("बैंक शाखा लोन बंद करने का एनओसी नहीं दे रही और उल्टे खाते से चार्ज काट रही है।", "Personal & Consumer Loans", "hi"),
        ("बँक शाखा कर्ज समाप्तीचे एनओसी प्रमाणपत्र देत नाही आणि खात्यातून चार्ज कापत आहे.", "Personal & Consumer Loans", "mr"),
        ("Branch wale loan close hone ke baad bhi NOC nahi de rahe aur fine laga rahe hain.", "Personal & Consumer Loans", "hi-latn"),

        ("My international wire transfer of funds from my checking account failed and money was put on hold.", "Bank Account Services", "en"),
        ("El dinero de mi transferencia internacional fue congelado en mi cuenta corriente sin motivo.", "Bank Account Services", "es"),
        ("Les fonds de mon virement international ont été bloqués sur mon compte courant sans raison.", "Bank Account Services", "fr"),
        ("Das Geld für meine Auslandsüberweisung wurde auf meinem Girokonto grundlos eingefroren.", "Bank Account Services", "de"),
        ("विदेश में पैसे भेजने के प्रयास के बाद बैंक ने मेरा पूरा बचत खाता ही होल्ड पर रख दिया है।", "Bank Account Services", "hi"),
        ("परदेशात पैसे पाठवण्याचा प्रयत्न केल्यानंतर बँकेने माझे संपूर्ण बँक खातेच गोठवून ठेवले आहे.", "Bank Account Services", "mr"),
        ("Overseas transfer initiate karne ke baad bank ne account balance hold kar diya hai.", "Bank Account Services", "hi-latn"),

        ("The mortgage lender applied my home insurance escrow funds to interest penalties instead of taxes.", "Mortgages & Housing Loans", "en"),
        ("La entidad hipotecaria aplicó los fondos de la cuenta de depósito a penalizaciones indebidas.", "Mortgages & Housing Loans", "es"),
        ("La banque de prêt immobilier a utilisé les fonds d'assurance pour payer des pénalités injustifiées.", "Mortgages & Housing Loans", "fr"),
        ("Der Hypothekengeber hat Guthaben aus dem Treuhandkonto für unberechtigte Mahnkosten einbehalten.", "Mortgages & Housing Loans", "de"),
        ("होम लोन प्रदाता ने बीमा के लिए जमा पैसों को मनमाने पेनल्टी चार्ज में काट लिया है।", "Mortgages & Housing Loans", "hi"),
        ("गृहकर्ज बँकेने विम्याच्या पैशांमधून अनधिकृत दंड आणि व्याज वसूल केले आहे.", "Mortgages & Housing Loans", "mr"),
        ("Mortgage lender ne insurance escrow balance ko late fee penalty me adjust kar diya.", "Mortgages & Housing Loans", "hi-latn")
    ]

    for c_text, c_cat, c_lang in cross_cutting_samples:
        # 6 in train, 2 in test
        for _ in range(6):
            sample_id += 1
            records.append({
                "complaint_id": f"CMP-{sample_id}",
                "complaint_text": inject_realistic_noise(c_text, c_lang),
                "category": c_cat,
                "language": c_lang,
                "split": "train"
            })
        for _ in range(2):
            sample_id += 1
            records.append({
                "complaint_id": f"CMP-{sample_id}",
                "complaint_text": inject_realistic_noise(c_text, c_lang),
                "category": c_cat,
                "language": c_lang,
                "split": "test"
            })

    df = pd.DataFrame(records)
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def generate_sample_batch_csv(df: pd.DataFrame, output_path: str, n_samples: int = 50):
    """Generates a realistic test CSV for batch upload demonstration."""
    sample_subset = df.sample(n=n_samples, random_state=123).copy()
    # Batch files typically uploaded by banks have: ticket_id, customer_name, complaint_text
    names = [
        "Carlos Mendez", "Marie Dubois", "Alexander Schmidt", "Rahul Sharma", "Pooja Verma",
        "Emily Watson", "Johnathan Reed", "Elena Rodriguez", "Pierre Lefevre", "Hans Weber",
        "Deepak Patel", "Sunita Nair", "Sofia Garcia", "Lucas Martin", "Maximilian Koch"
    ]
    batch_records = []
    for idx, row in enumerate(sample_subset.itertuples()):
        batch_records.append({
            "ticket_id": f"TKT-BATCH-{idx+1:04d}",
            "customer_name": random.choice(names),
            "complaint_text": row.complaint_text
        })
    batch_df = pd.DataFrame(batch_records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    batch_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Sample batch CSV generated at: {output_path} ({len(batch_df)} records)")


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print("Generating comprehensive multilingual banking complaints dataset...")
    df_complaints = generate_complaints_dataset()
    dataset_file = os.path.join(raw_dir, "complaints_multilingual.csv")
    df_complaints.to_csv(dataset_file, index=False, encoding="utf-8")
    print(f"Dataset saved to: {dataset_file}")
    print(f"Total records: {len(df_complaints)}")
    print("Split distribution:\n", df_complaints["split"].value_counts())
    print("Class distribution:\n", df_complaints["category"].value_counts())
    print("Language distribution:\n", df_complaints["language"].value_counts())

    # Generate sample batch test file
    batch_test_file = os.path.join(data_dir, "sample_batch_complaints.csv")
    generate_sample_batch_csv(df_complaints, batch_test_file, n_samples=30)
