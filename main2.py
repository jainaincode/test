from fastapi import FastAPI, Request, Form, HTTPException, Depends, Header
from fastapi.responses import HTMLResponse, JSONResponse, Response, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel
from dotenv import load_dotenv
import os, json, mysql.connector, uuid
from werkzeug.security import check_password_hash
from datetime import datetime, timedelta
from typing import Optional
from decimal import Decimal
from pet_agent import create_pet_agent
from logger import logger
from landing_page_agent import create_landing_agent
from summary import create_summary
from emailer import send_email_jency, send_roi_email

# from pet_agent2 import create_pet_agent2
# from langgraph_supervisor import create_pet_agent3


load_dotenv()

app = FastAPI()

# Session + CORS
app.add_middleware(SessionMiddleware, secret_key="SECRET-KEY")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates & Static
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# DB Config
MYSQL_USER = os.getenv("DB_USER")
MYSQL_PASSWORD = os.getenv("DB_PASSWORD")
MYSQL_HOST = os.getenv("DB_HOST")
MYSQL_DB = os.getenv("DB_NAME")

mysql_config = {
    "user": MYSQL_USER,
    "password": MYSQL_PASSWORD,
    "host": MYSQL_HOST,
    "database": MYSQL_DB,
}

SESSION_EXPIRY_DAYS = 1


def get_db():
    conn = mysql.connector.connect(**mysql_config)
    cursor = conn.cursor(dictionary=True)
    return conn, cursor


# Models
class LoginRequest(BaseModel):
    email: str
    password: str


class QueryRequest(BaseModel):
    user_query: str
    session_id: Optional[str] = None


class MakeQuery(BaseModel):
    query: str = None
    img_desc: str = None


class SummaryRequest(BaseModel):
    session_id: str


# Utils
def is_pet_profile_complete(user_id: int):
    conn, cursor = get_db()
    cursor.execute(
        """
        SELECT *
        FROM pets
        WHERE user_id = %s
    """,
        (user_id,),
    )
    profiles = cursor.fetchall()
    cleaned_profiles = []
    for profile in profiles:
        cleaned_profile = {}
        for key, value in profile.items():
            if isinstance(value, Decimal):
                cleaned_profile[key] = float(value)
            elif isinstance(value, datetime):
                cleaned_profile[key] = value.isoformat() if value else None
            else:
                cleaned_profile[key] = value
        cleaned_profiles.append(cleaned_profile)
    # print("profiles", cleaned_profiles)
    conn.close()
    if cleaned_profiles == []:
        return True, cleaned_profiles
    return False, cleaned_profiles


def get_user_id(request: Request, x_user_id: Optional[int] = None):
    user_id = request.session.get("user_id")
    if not user_id and x_user_id:
        user_id = x_user_id
    return user_id


def get_reviews(product_id: int):
    conn, cursor = get_db()
    try:
        cursor.execute(
            """
            SELECT rating,review_text,photo_url FROM reviews
            WHERE product_id = %s AND is_delete = 0
            ORDER BY created_at DESC
            """,
            (product_id,),
        )
        reviews = cursor.fetchall()
        cleaned_reviews = []
        for review in reviews:
            cleaned_reviews.append(
                {
                    "rating": (
                        float(review["rating"])
                        if isinstance(review["rating"], Decimal)
                        else review["rating"]
                    ),
                    "review_text": review["review_text"],
                    "photo_url": review["photo_url"],
                }
            )

        return {"reviews": cleaned_reviews}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching reviews: {e}")
    finally:
        conn.close()


def get_product_id_by_name(product_name: str) -> Optional[int]:
    conn, cursor = get_db()
    cursor.execute("SELECT product_id FROM products WHERE name = %s", (product_name,))
    row = cursor.fetchone()
    logger.info(f"Inside get_product_id_by_name: {row}")
    conn.close()
    return row[0] if row else None


def store_conversation(
    user_id, user_mail, message, response, session_id, conversation_id
):
    db, cursor = get_db()
    cursor.execute(
        """
        INSERT INTO clyro_agent_chat_logs 
        (user_id, user_email, session_id, message, ai_response, timestamp,conversation_id, platform)
        VALUES (%s, %s, %s, %s, %s, %s,%s, "web")
        """,
        (
            user_id,
            user_mail,
            session_id,
            message,
            response,
            datetime.now(),
            conversation_id,
        ),
    )
    db.commit()
    db.close()


def get_user_id_from_mail(user_mail: str):
    conn, cursor = get_db()
    cursor.execute(
        "SELECT user_id FROM users WHERE email = %s AND is_delete = 0", (user_mail,)
    )
    row = cursor.fetchone()
    conn.close()
    return row["user_id"] if row else None


def get_user_mail_by_id(user_id: int):
    conn, cursor = get_db()
    cursor.execute(
        "SELECT email FROM users WHERE user_id = %s AND is_delete = 0", (user_id,)
    )
    row = cursor.fetchone()
    conn.close()
    return row["email"] if row else None


# Function to get last session
def get_last_session(user_id):
    """Fetch the most recent session for the user."""
    conn, cursor = get_db()
    cursor.execute(
        """
        SELECT * FROM clyro_chat_sessions
        WHERE user_id = %s
        ORDER BY start_time DESC
        LIMIT 1
        """,
        (user_id,),
    )
    session = cursor.fetchone()
    conn.close()
    return session


def create_session(user_mail: str):
    session_id = str(uuid.uuid4())
    user_id = get_user_id_from_mail(user_mail)

    conn, cursor = get_db()
    cursor.execute(
        """
        INSERT INTO clyro_chat_sessions (session_id, user_id, user_email, start_time, is_active)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (session_id, user_id, user_mail, datetime.now(), True),
    )
    conn.commit()
    cursor.close()
    conn.close()

    return session_id


# This function is for condition when user is not logged in and still wants to chat with bot
def create_guest_session():
    session_id = str(uuid.uuid4())

    conn, cursor = get_db()
    cursor.execute(
        """
        INSERT INTO clyro_chat_sessions (session_id, user_id, user_email, start_time, is_active, is_guest)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (session_id, None, None, datetime.now(), True, True),
    )
    conn.commit()
    cursor.close()
    conn.close()

    return session_id


# Function to create new conversation id:
def create_conversation(session_id, user_id):
    conversation_id = str(uuid.uuid4())
    conn, cursor = get_db()
    cursor.execute(
        """
        INSERT INTO clyro_conversation_session
        (conversation_id, session_id, user_id, start_time, status,store_id)
        VALUES (%s, %s, %s, %s, 'active','001')
        """,
        (conversation_id, session_id, user_id, datetime.now()),
    )
    conn.commit()
    conn.close()
    return conversation_id


def end_conversation(conversation_id):
    conn, cursor = get_db()
    cursor.execute(
        """
        UPDATE clyro_conversation_session
        SET end_time = %s, status = 'ended'
        WHERE conversation_id = %s
        """,
        (
            datetime.now(),
            conversation_id,
        ),
    )
    conn.commit()
    conn.close()


def get_last_conversation(session_id):
    conn, cursor = get_db()
    cursor.execute(
        """
        SELECT c.conversation_id, MAX(l.timestamp) as last_msg_time
        FROM clyro_conversation_session c
        LEFT JOIN clyro_agent_chat_logs l
        ON c.conversation_id = l.conversation_id
        WHERE c.session_id = %s
        GROUP BY c.conversation_id
        ORDER BY last_msg_time DESC
        LIMIT 1

        """,
        (session_id,),
    )
    row = cursor.fetchone()
    print("last conversation row", row)
    conn.close()
    return row


def resolve_conversation(session_id, user_id):
    last_convo = get_last_conversation(session_id)

    if not last_convo:
        # No active conversation → create new
        return create_conversation(session_id, user_id)

    last_time = last_convo["last_msg_time"]
    convo_id = last_convo["conversation_id"]

    # Keep conversation active if no messages yet
    if last_time is None:
        return convo_id

    # Only expire if 30 minutes passed since last message
    if datetime.now() - last_time >= timedelta(minutes=30):
        print(
            "Expiring conversation due to 30 min inactivity", datetime.now(), last_time
        )
        end_conversation(convo_id)
        return create_conversation(session_id, user_id)

    # Otherwise, keep the same conversation
    return convo_id


"""one = get_product_id_by_name("royal canin medium adult")
print("one", one)"""


# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/login")
async def api_login(data: LoginRequest, request: Request):
    email = data.email
    password = data.password

    # print(email, password)
    logger.info(f"Inside api/login: email={email}")

    if not (email and password):
        raise HTTPException(status_code=400, detail="Missing email or password")

    conn, cursor = get_db()
    cursor.execute("SELECT * FROM users WHERE email = %s AND is_delete = 0", (email,))
    user = cursor.fetchone()
    logger.info("Inside api/login")
    if not user:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")

    if not check_password_hash(user["password_hash"], password):
        print("error in password")
        conn.close()
        raise HTTPException(status_code=401, detail="Incorrect password")

    # Update last_login timestamp
    now = datetime.now()
    cursor.execute(
        "UPDATE users SET last_login = %s WHERE user_id = %s", (now, user["user_id"])
    )
    conn.commit()

    # Store in session
    session_id = create_session(data.email)
    logger.info(f"session_id created: {session_id} for user {data.email}")

    request.session["user_id"] = user["user_id"]
    request.session["user_name"] = user["name"]
    request.session["user_mail"] = data.email

    cursor.close()
    conn.close()

    return JSONResponse(
        {
            "success": True,
            "message": "Login successful",
            "user_id": user["user_id"],
            "name": user["name"],
            "session_id": session_id,
        }
    )


@app.get("/api/pet-profile-status")
async def pet_profile_status(request: Request, x_user_id: Optional[int] = Header(None)):
    user_id = get_user_id(request, x_user_id)
    complete = is_pet_profile_complete(user_id)
    return {"profile_complete": complete}


@app.get("/api/pet-profiles")
async def pet_profiles(request: Request, x_user_id: Optional[int] = Header(None)):
    user_id = get_user_id(request, x_user_id)
    conn, cursor = get_db()
    cursor.execute(
        """
        SELECT *
        FROM pets
        WHERE user_id = %s
        AND is_delete = 0
        """,
        (user_id,),
    )
    rows = cursor.fetchall()

    conn.close()

    return JSONResponse(content=jsonable_encoder({"pets": rows}))


@app.get("/api/pet-journals")
async def get_pet_journals(request: Request, x_user_id: Optional[int] = Header(None)):
    user_id = get_user_id(request, x_user_id)

    if not user_id:
        raise HTTPException(status_code=401, detail="User not authenticated")

    conn, cursor = get_db()

    # Step 1: Get all active pets for the user
    cursor.execute(
        """
        SELECT pet_id FROM pets
        WHERE user_id = %s AND is_delete = 0
        """,
        (user_id,),
    )
    pet_ids = [row["pet_id"] for row in cursor.fetchall()]

    if not pet_ids:
        conn.close()
        return JSONResponse(content={"journals": []})

    # Step 2: Get all journal entries for these pets
    format_strings = ",".join(["%s"] * len(pet_ids))
    cursor.execute(
        f"""
        SELECT pet_id, entry_date, entry_type, details
        FROM pet_journals
        WHERE pet_id IN ({format_strings}) AND is_delete = 0
        ORDER BY entry_date DESC
        """,
        tuple(pet_ids),
    )

    journal_entries = cursor.fetchall()
    logger.info("Inside api/pet-journals and journal_entries fetched")
    conn.close()

    return JSONResponse(content={"journals": journal_entries})


@app.get("/chatbot", response_class=HTMLResponse)
async def chatbot(request: Request, x_user_id: Optional[int] = Header(None)):
    user_id = get_user_id(request, x_user_id)

    if not user_id:
        return templates.TemplateResponse(
            "chatbot.html", {"request": request, "profile_complete": False}
        )
    complete = is_pet_profile_complete(user_id)
    # print(user_id)
    # print(complete)
    request.session["chatbot_name"] = "Pet_store"
    request.session["pet_profile_complete"] = complete
    return templates.TemplateResponse(
        "chatbot.html", {"request": request, "profile_complete": complete}
    )


# this is for langchain
@app.post("/get_response")
async def get_response(
    request: Request, data: QueryRequest, x_user_id: Optional[int] = Header(None)
):
    user_id = get_user_id(request, x_user_id)
    user_mail = request.session.get("user_mail") or get_user_mail_by_id(user_id)
    session_id = data.session_id
    # print(session_id, "session_id")
    complete = is_pet_profile_complete(user_id)
    profile_complete = complete

    session_expire = False

    if user_mail:  # logged-in user
        if session_id is None:
            last_session = get_last_session(user_id)

            if last_session:
                last_time = last_session["start_time"]
                if datetime.now() - last_time > timedelta(days=SESSION_EXPIRY_DAYS):
                    # Expired session → create new
                    session_id = create_session(user_mail)
                    session_expire = True
                    logger.info(f"Expired session → new session created: {session_id}")
                else:
                    # Reuse active session
                    session_id = last_session["session_id"]
                    logger.info(f"Resumed active session: {session_id}")
            else:
                # No previous session → create new
                session_id = create_session(user_mail)
                session_expire = True
                logger.info(f"No previous session → new session created: {session_id}")
        else:
            logger.info(f"Using provided session_id for logged-in user: {session_id}")

    else:  # guest user
        if session_id is None:
            session_id = create_guest_session()
            logger.info(f"Guest session created: {session_id}")
        else:
            logger.info(f"Using provided guest session_id: {session_id}")

    conversation_id = resolve_conversation(session_id, user_id)
    print("conversation_id", conversation_id)
    try:
        logger.info(f"Session ID in get_response in try: {session_id}")
        pet_agent = create_pet_agent()
        if not user_id:
            combined_input = f"{data.user_query}"
        combined_input = f"{data.user_query} | Pet Profile Needed: {profile_complete} | user_id : {user_id}"
        # print(combined_input)
        response = pet_agent.invoke(
            {"input": combined_input},
            config={"configurable": {"session_id": session_id or "chatbot_name"}},
        )
        # print(response["output"])

        output_json = json.loads(response["output"])
        if output_json.get("type") == "review":
            product_id = output_json.get("product_id")
            if not product_id:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Missing product_id for review type"},
                )

            review_data = get_reviews(product_id=product_id)
            if review_data == {"reviews": []}:
                review_data = "Sorry, there are no reviews present at the moment for this product."

            store_conversation(
                user_id,
                user_mail,
                data.user_query,
                json.dumps(review_data),
                session_id,
                conversation_id,
            )
            logger.info("Agent response in get_response 200 OK")
            return JSONResponse(
                content={
                    "response": review_data,
                    "session_id": session_id,
                    "session_expire": session_expire,
                }
            )

        # -------- Handle cart type --------
        elif output_json.get("type") == "cart":
            product_ids = output_json.get("product_id")
            pet_ids = output_json.get("pet_id")
            quantities = output_json.get("quantity")

            # Ensure lists
            if not isinstance(product_ids, list):
                product_ids = [product_ids]
            if not isinstance(pet_ids, list):
                pet_ids = [pet_ids]
            if not isinstance(quantities, list):
                quantities = [quantities]

            product_ids = [int(pid) for pid in product_ids]
            pet_ids = [int(pid) for pid in pet_ids]
            quantities = [int(q) for q in quantities]

            # --- Convert product IDs safely ---
            resolved_product_ids = []
            for pid in product_ids:
                try:
                    resolved_product_ids.append(int(pid))  # if already int
                except ValueError:
                    # fallback: lookup in DB by name
                    db_id = get_product_id_by_name(pid)
                    if not db_id:
                        return JSONResponse(
                            status_code=400,
                            content={"error": f"Invalid product identifier: {pid}"},
                        )
                    resolved_product_ids.append(db_id)

            # Convert pet_ids & quantities to int
            pet_ids = [int(p) for p in pet_ids]
            quantities = [int(q) for q in quantities]

            # Call order function directly
            await place_order(
                user_id=user_id,
                delivery_address=output_json.get("delivery_address", ""),
                payment_method=output_json.get("payment_method", "COD") or "online",
                special_notes="na",
                product_id=product_ids,
                pet_id=pet_ids,
                quantity=quantities,
            )

            store_conversation(
                user_id,
                user_mail,
                data.user_query,
                json.dumps(output_json),
                session_id,
                conversation_id,
            )
            logger.info("Agent response in orders")
            return JSONResponse(
                content={
                    "response": output_json,
                    "session_id": session_id,
                    "session_expire": session_expire,
                }
            )

        logger.info("Agent response in get_response 200 OK")
        store_conversation(
            user_id,
            user_mail,
            data.user_query,
            json.dumps(output_json),
            session_id,
            conversation_id,
        )
        # print("stored")
        logger.info("Stored in database")
        return JSONResponse(
            content={
                "response": output_json,
                "session_id": session_id,
                "session_expire": session_expire,
            }
        )
    except Exception as e:
        logger.info(f"Agent response Error {e}")
        return Response(f"Error during generation: {e}", media_type="text/plain")


@app.get("/api/conversations/{user_id}")
async def get_conversations(user_id: int):
    try:
        conn, cursor = get_db()
        cursor.execute(
            "SELECT * FROM clyro_agent_chat_logs WHERE user_id = %s ORDER BY timestamp ASC",
            (user_id,),
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        logger.info("Inside api/conversations/{user_id}")
        if not rows:
            return {"success": True, "conversations": []}

        return {"success": True, "conversations": rows}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/session-summary")
async def session_summary(data: SummaryRequest):
    session_id = data.session_id
    # logger.info("session_id", session_id)
    conn, cursor = get_db()
    try:
        # 1. Check if session exists
        cursor.execute(
            "SELECT * FROM clyro_chat_sessions WHERE session_id = %s", (session_id,)
        )
        session = cursor.fetchone()
        logger.info(f"row: {session}")
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        else:
            print("Session found", session)

        # 2. Find last summary end_time
        cursor.execute(
            "SELECT end_time FROM clyro_session_summaries WHERE session_id = %s ORDER BY end_time DESC LIMIT 1",
            (session_id,),
        )
        last_summary = cursor.fetchone()
        last_end_time = (
            last_summary["end_time"] if last_summary else session["start_time"]
        )

        # 3. Get chat logs between last_end_time and now
        now = datetime.now()
        cursor.execute(
            """SELECT message, ai_response FROM clyro_agent_chat_logs WHERE session_id = %s AND timestamp > %s AND timestamp <= %s
            ORDER BY timestamp ASC""",
            (session_id, last_end_time, now),
        )
        all_rows = cursor.fetchall()
        # print("all_rows", all_rows)
        conversations = []
        for row in all_rows:
            # print("jency", row)

            user_msg = row.get("message") or ""
            bot_msg = ""

            try:
                ai_resp = row.get("ai_response")
                if ai_resp:
                    bot_json = json.loads(ai_resp)
                    bot_msg = bot_json.get("text") or str(bot_json)
                else:
                    bot_msg = ""
            except json.JSONDecodeError:
                try:
                    bot_json = json.loads(json.loads(ai_resp))
                    bot_msg = bot_json.get("text") or str(bot_json)
                except Exception:
                    bot_msg = ai_resp or ""

            conversations.append((user_msg, bot_msg))

        logger.info(f"Final conversations:{conversations}")

        # 3. Generate summary
        summary_obj = create_summary(conversations)
        # print("Generated summary (raw):", summary_obj, type(summary_obj))

        # Ensure it’s stored as JSON string
        summary_text = json.dumps(summary_obj)
        logger.info(f"Summary text (JSON string):{summary_text}")

        # 4. Save summary in DB
        cursor.execute(
            """INSERT INTO clyro_session_summaries (session_id, summary, start_time, end_time)
            VALUES (%s, %s, %s, %s)""",
            (session_id, summary_text, last_end_time, now),
        )
        conn.commit()

        return JSONResponse(
            content={
                "success": True,
                "message": "Summary generated and saved",
                "session_id": session_id,
            }
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"[ERROR] session_summary: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})
    finally:
        conn.close()


# this api is to get summary
@app.get("/api/get-summary/{session_id}")
async def get_summary(session_id: str):
    conn, cursor = get_db()
    try:
        cursor.execute(
            """SELECT summary FROM clyro_session_summaries 
                WHERE session_id = %s 
                ORDER BY end_time DESC 
                LIMIT 1
            """,
            (session_id,),
        )
        row = cursor.fetchone()
        if not row or not row.get("summary"):
            raise HTTPException(status_code=404, detail="Summary not found")

        summary_obj = json.loads(row["summary"])

        return JSONResponse(
            content={
                "response": summary_obj,
                "session_id": session_id,
                "success": True,
            }
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"[ERROR] get_summary: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})
    finally:
        conn.close()


# API for bokking demo form in landing page
@app.post("/book-demo")
async def book_demo(
    name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    message: str = Form(...),
):
    """Receive demo-booking form data and email it to prerna@narola.ai."""
    body = (
        f"New demo booking details:\n\n"
        f"Name: {name}\n"
        f"Email: {email}\n"
        f"Phone: {phone}\n"
        f"Message: {message}"
    )

    # use the common emailer function
    send_email_jency(
        receiver="Prerna@narola.ai", subject="Demo booking for Clyro", body=body
    )

    return {"status": "success", "message": "Demo booked and email sent to Prerna."}


# API to send ROI result email
@app.post("/SendROIResult")
def send_roi_result(
    email: str = Form(...),
    monthlyVisitors: int = Form(...),
    averageOrderValue: float = Form(...),
    conversionRate: float = Form(...),
    includeSupportSavings: bool = Form(...),
    totalMonthlyROI: float = Form(...),
    annualROI: float = Form(...),
    currentMonthlyRevenue: float = Form(...),
    salesLiftDelta: float = Form(...),
    orderValueDelta: float = Form(...),
    supportCostSavings: float = Form(...),
):
    try:
        # 1️⃣ Email to Prerna
        prerna_html = f"""
        <html>
        <body>
            <h2>New ROI Calculator Submission</h2>
            <p><strong>User Email:</strong> {email}</p>
            <p><strong>Monthly Visitors:</strong> {monthlyVisitors:,}</p>
            <p><strong>AOV:</strong> ${averageOrderValue}</p>
            <p><strong>Baseline Conversion Rate:</strong> {conversionRate}%</p>
            <p><strong>Include Support Savings:</strong> {'Yes' if includeSupportSavings else 'No'}</p>
            <hr/>
            <h3>Calculated Results</h3>
            <ul>
            <li><strong>Estimated Monthly ROI:</strong> ${totalMonthlyROI:,.0f}</li>
            <li><strong>Annual ROI:</strong> ${annualROI:,.0f}</li>
            <li><strong>Current Monthly Revenue:</strong> ${currentMonthlyRevenue:,.0f}</li>
            <li><strong>+15% Potential New Sales:</strong> +${salesLiftDelta:,.0f}</li>
            <li><strong>+10% Growth in Order Value:</strong> +${orderValueDelta:,.0f}</li>
            <li><strong>-30% Support Costs:</strong> ${supportCostSavings:,.0f} saved</li>
            </ul>
        </body>
        </html>
        """
        send_roi_email(
            "Prerna@narola.ai",
            "New ROI Calculator Submission",
            prerna_html,
            reply_to=email,
        )

        # 2️⃣ Confirmation Email to User
        user_html = f"""
        <html>
        <body>
            <h2>Your Clyro ROI Estimate</h2>
            <p>Thanks for trying the ROI calculator! Here's your summary:</p>
            <p><strong>Monthly Visitors:</strong> {monthlyVisitors:,}</p>
            <p><strong>AOV:</strong> ${averageOrderValue}</p>
            <p><strong>Baseline Conversion:</strong> {conversionRate}%</p>
            <hr/>
            <p><strong>Estimated Monthly ROI:</strong> ${totalMonthlyROI:,.0f}</p>
            <ul>
            <li><strong>+15% Potential New Sales:</strong> +${salesLiftDelta:,.0f}</li>
            <li><strong>+10% Growth in Order Value:</strong> +${orderValueDelta:,.0f}</li>
            <li><strong>-30% Support Costs:</strong> ${supportCostSavings:,.0f} saved</li>
            </ul>
            <p>Want us to apply this to your store? Just reply to this email and we'll set up a quick demo.</p>
        </body>
        </html>
        """
        send_roi_email(
            email, "Your Clyro ROI estimate", user_html, reply_to="Prerna@narola.ai"
        )

        return {"status": "emails sent"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# API for video access
@app.post("/SendVideoAccessRequest")
def send_video_access_request(email: str = Form(...)):
    try:
        html_body = f"""
        <html>
          <body>
            <h2>New Video Access Request</h2>
            <p>A user has requested to watch the concept video.</p>
            <p><strong>Email:</strong> {email}</p>
            <p style="color:#64748b">Submitted at: </p>
          </body>
        </html>
        """
        send_roi_email(
            "Prerna@narola.ai",
            f"New Video Access Request from {email}",
            html_body,
            reply_to=email,
        )
        return {"status": "email sent"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""
#this is for langgraph
@app.post("/get_response")
async def get_response(
    request: Request, data: QueryRequest, x_user_id: Optional[int] = Header(None)
):
    user_id = get_user_id(request, x_user_id)
    pet_profile_needed, profiles = is_pet_profile_complete(user_id)

    pet_profile_needed = not pet_profile_needed
    if profiles:
        pet_data = profiles[0]
        pet_id = pet_data["pet_id"]
        pet_context = (
            f"Pet Name: {pet_data['name']}, Species: {pet_data['species']}, "
            f"Breed: {pet_data['breed']}, Date of Birth: {pet_data['date_of_birth'] or 'unknown'}, "
            f"Gender: {pet_data['gender']}, Weight: {pet_data['weight_kg']} kg, "
            f"Allergies: {pet_data['allergies'] or 'none'}, "
            f"Adoption Date: {pet_data['adoption_date'] or 'unknown'}, "
            f"Current Medication: {pet_data['current_medication'] or 'none'}, "
            f"Milestones: {pet_data['milestones'] or 'none'}, "
            f"Notes: {pet_data['notes'] or 'none'}"
        )

    # Retrieve and reset collected_fields from session for the current turn
    collected_fields = request.session.get("collected_fields", {})
    # Deduplicate and limit chat history
    history = request.session.get("chat_history", [])
    # Remove duplicates while preserving order
    seen = set()
    deduped_history = []
    for msg in history:
        msg_tuple = (msg["type"], msg["content"])
        if msg_tuple not in seen:
            seen.add(msg_tuple)
            deduped_history.append(msg)
    # Limit to last 10 unique messages
    history = deduped_history[-10:]
    # Convert to LangChain message objects
    history_messages = []
    for msg in history:
        if msg["type"] == "human":
            history_messages.append(HumanMessage(content=msg["content"]))
        elif msg["type"] == "ai":
            history_messages.append(AIMessage(content=msg["content"]))

    try:
        pet_agent = create_pet_agent3()
        # Fix combined_input to use correct pet_profile_needed
        combined_input = f"{data.user_query} | Pet Profile Needed: {pet_profile_needed} | user_id: {user_id if user_id else 'None'} | pet_id: {pet_id if pet_id else 'None'}"

        initial_state = {
            "messages": history_messages + [HumanMessage(content=combined_input)],
            "user_id": user_id,
            "pet_id": pet_id,
            "pet_context": pet_context,
            "pet_profile_needed": pet_profile_needed,
            "collected_fields": collected_fields,
            "response": {},
            "intermediate_steps": [],
            "next_agent": "supervisor",
            "task_complete": False,
            "current_task": data.user_query,
            "is_logged_in": bool(user_id),
            "is_final_response": False,
        }

        final_state = pet_agent.invoke(
            initial_state,
            config={"configurable": {"session_id": "chatbot_name"}},
        )
        print("jency")
        print(f"Agent final state: {final_state}")

        last_message = final_state["messages"][-1].content

        try:
            output_json = json.loads(last_message)
        except json.JSONDecodeError:
            output_json = {"type": "text", "text": last_message}

        # Ensure collected_fields is JSON-serializable
        cleaned_collected_fields = {}
        for key, value in final_state.get("collected_fields", {}).items():
            if isinstance(value, Decimal):
                cleaned_collected_fields[key] = float(value)
            elif isinstance(value, datetime):
                cleaned_collected_fields[key] = value.isoformat() if value else None
            else:
                cleaned_collected_fields[key] = value
        request.session["collected_fields"] = cleaned_collected_fields

        if output_json.get(
            "type"
        ) == "text" and "successfully logged in" in output_json.get("text", ""):
            conn, cursor = get_db()
            email = cleaned_collected_fields.get("email")
            if email:
                cursor.execute(
                    "SELECT user_id FROM users WHERE email = %s AND is_delete = 0",
                    (email,),
                )
                user = cursor.fetchone()
                if user:
                    request.session["user_id"] = user["user_id"]
            conn.close()
            request.session.pop("collected_fields", None)

        if "successfully created" in output_json.get("text", ""):
            request.session.pop("collected_fields", None)

        # Update chat history with deduplicated entries
        history.append({"type": "human", "content": data.user_query})
        history.append({"type": "ai", "content": output_json.get("text", last_message)})
        # Deduplicate again before saving
        seen = set()
        deduped_history = []
        for msg in history:
            msg_tuple = (msg["type"], msg["content"])
            if msg_tuple not in seen:
                seen.add(msg_tuple)
                deduped_history.append(msg)
        request.session["chat_history"] = deduped_history[-10:]

        if output_json.get("type") == "review":
            product_id = output_json.get("product_id")
            if not product_id:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Missing product_id for review type"},
                )

            review_data = get_reviews(product_id=product_id)
            if review_data == {"reviews": []}:
                return JSONResponse(
                    content={
                        "type": "text",
                        "text": "Sorry, there are no reviews present at the moment for this product. Would you like to see similar products or bundles?",
                    }
                )

            output_json.update({"reviews": review_data["reviews"]})

        return JSONResponse(content=output_json)

    except Exception as e:
        print(f"Error during LangGraph invocation: {e}")
        return Response(
            f"Error during generation: {e}", media_type="text/plain", status_code=500
        )

"""


@app.api_route("/process-query-make", methods=["GET", "POST"])
async def process_make_query(request: Request):
    if request.method == "POST":
        data = await request.json()
        user_query = data.get("query") or data.get("img_desc")
        pet_agent = create_pet_agent()
        response = pet_agent.invoke(
            {"input": user_query},
            config={"configurable": {"session_id": "chatbot_name"}},
        )
        response_text = response["output"]
        json_start = response_text.find("{")
        json_end = response_text.rfind("}")
        if json_start != -1 and json_end != -1:
            try:
                json_str = response_text[json_start : json_end + 1]
                parsed = json.loads(json_str)
                return JSONResponse(content=parsed)
            except json.JSONDecodeError:
                return JSONResponse(content={"error": "Invalid JSON response"})
        return JSONResponse(content={"response": "No valid JSON response found"})
    return {"response": "Hi, in GET"}


# ---------- Clyro Landing Page APIs ----------
def store_conversation_landing_page(user_mail, message, response):
    db, cursor = get_db()
    cursor.execute(
        "INSERT INTO clyro_admin_chat_logs (user_mail, message, response, timestamp) VALUES (%s, %s, %s, %s)",
        (user_mail, message, response, datetime.now()),
    )
    db.commit()


@app.post("/get_response_for_landing_page")
async def get_response_for_landing_page(request: Request, data: QueryRequest):
    user_query = data.user_query
    logger.info(f"User Query in landing page:{user_query}")
    try:
        # Create agent
        clyro_agent = create_landing_agent()

        # Invoke agent
        response = clyro_agent.invoke(
            {"input": user_query},
            config={"configurable": {"session_id": "chatbot_name"}},
        )

        # Extract output safely
        raw_output = response.get("output")

        parsed_response = None

        # Case 1: Already dict
        if isinstance(raw_output, dict):
            parsed_response = raw_output

        # Case 2: String output, try parsing JSON inside
        elif isinstance(raw_output, str):
            json_start = raw_output.find("{")
            json_end = raw_output.rfind("}")
            if json_start != -1 and json_end != -1:
                json_str = raw_output[json_start : json_end + 1]
                try:
                    parsed_response = json.loads(json_str)
                except json.JSONDecodeError as decode_err:
                    return Response(
                        f"JSON parsing error: {decode_err}", media_type="text/plain"
                    )

        if not parsed_response:
            return Response("No valid JSON response found", media_type="text/plain")

        # ---- Handle email (lead) ----
        if "lead" in parsed_response:
            # Save email into session for this user
            request.session["user_mail"] = parsed_response["lead"]

        # ---- Always store conversation ----
        user_mail = request.session.get("user_mail", None)
        if user_mail:
            # Save user query + bot response in DB
            store_conversation_landing_page(
                user_mail, user_query, json.dumps(parsed_response)
            )

        return JSONResponse(content=parsed_response)

    except Exception as e:
        return Response(f"Error during generation: {e}", media_type="text/plain")


# ---------- Jainain's Code ---------------------

# amFpbmFpbg==
from fastapi import FastAPI, Form, Request, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from datetime import datetime, timedelta
import requests

from helper import (
    get_db_connection,
    hash_password,
    verify_password,
    fetch_all,
    fetch_one,
    execute_query,
    execute_many,
    send_email,
)


# -------- PRODUCTS --------
@app.get("/products")
def get_products():
    return fetch_all("SELECT * FROM products WHERE is_delete = 0")


@app.get("/products/{product_id}")
def get_product_by_id(product_id: int):
    row = fetch_one(
        "SELECT * FROM products WHERE product_id = %s AND is_delete = 0", (product_id,)
    )
    if row:
        return row
    return JSONResponse({"error": "Product not found"}, status_code=404)


# -------- SIGNUP --------
@app.post("/signup")
def signup(
    name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    password: str = Form(...),
):
    existing_user = fetch_all(
        "SELECT user_id FROM users WHERE email = %s AND is_delete = 0", (email,)
    )
    if existing_user:
        return JSONResponse(
            {"error": "A user with this email already exists"}, status_code=409
        )

    password_hash = hash_password(password)
    now = datetime.now()
    execute_query(
        """
        INSERT INTO users (
            name, email, phone, password_hash,
            signup_date, last_login, preferred_channel,
            created_date, modified_date, is_delete
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """,
        (name, email, phone, password_hash, now, now, "Email", now, now, 0),
        commit=True,
    )

    return JSONResponse({"message": "User registered successfully"}, status_code=201)


# -------- LOGIN --------
@app.post("/login")
def login(email: str = Form(...), password: str = Form(...)):
    user = fetch_one("SELECT * FROM users WHERE email = %s AND is_delete = 0", (email,))
    if not user:
        return JSONResponse({"error": "User not found"}, status_code=404)
    if not verify_password(user["password_hash"], password):
        return JSONResponse({"error": "Incorrect password"}, status_code=401)

    execute_query(
        "UPDATE users SET last_login = %s WHERE user_id = %s",
        (datetime.now(), user["user_id"]),
        commit=True,
    )
    return {
        "message": "Login successful",
        "user_id": user["user_id"],
        "name": user["name"],
    }


# -------- PETS --------
@app.post("/pets")
async def add_pet(request: Request):
    data = await (
        request.json()
        if request.headers.get("content-type") == "application/json"
        else request.form()
    )
    required_fields = [
        "user_id",
        "name",
        "species",
        "breed",
        "date_of_birth",
        "gender",
        "weight_kg",
        "photo_url",
        "allergies",
        "notes",
        "adoption_date",
    ]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return JSONResponse(
            {"error": f"Missing fields: {', '.join(missing)}"}, status_code=400
        )

    execute_query(
        """
        INSERT INTO pets (
            user_id, name, species, breed, date_of_birth,
            gender, weight_kg, photo_url, allergies,
            notes, adoption_date, created_date, modified_date, is_delete
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 0)
    """,
        (
            data["user_id"],
            data["name"],
            data["species"],
            data["breed"],
            data["date_of_birth"],
            data["gender"],
            data["weight_kg"],
            data["photo_url"],
            data["allergies"],
            data["notes"],
            data["adoption_date"],
            datetime.now(),
            datetime.now(),
        ),
        commit=True,
    )

    return JSONResponse({"message": "Pet added successfully"}, status_code=201)


@app.get("/user_pets")
def get_user_pets(user_id: int = Query(...)):
    return fetch_all(
        "SELECT * FROM pets WHERE user_id = %s AND is_delete = 0", (user_id,)
    )


@app.get("/pets/{pet_id}")
def get_pet_by_id(pet_id: int):
    pet = fetch_one("SELECT * FROM pets WHERE pet_id = %s AND is_delete = 0", (pet_id,))
    if pet:
        return pet
    return JSONResponse({"error": "Pet not found"}, status_code=404)


@app.put("/pets/{pet_id}")
async def update_pet(pet_id: int, request: Request):
    data = await request.form()
    allowed_fields = [
        "name",
        "species",
        "breed",
        "date_of_birth",
        "gender",
        "weight_kg",
        "photo_url",
        "allergies",
        "notes",
        "adoption_date",
        "last_journal_update",
    ]
    updates, values = [], []
    for field in allowed_fields:
        if field in data:
            updates.append(f"{field} = %s")
            values.append(data[field])
    if not updates:
        return JSONResponse({"error": "No fields to update"}, status_code=400)

    updates.append("modified_date = %s")
    values.append(datetime.now())
    values.append(pet_id)

    execute_query(
        f"UPDATE pets SET {', '.join(updates)} WHERE pet_id = %s AND is_delete = 0",
        tuple(values),
        commit=True,
    )
    return {"message": "Pet updated successfully"}


@app.delete("/pets/{pet_id}")
def delete_pet(pet_id: int):
    execute_query(
        "UPDATE pets SET is_delete = 1, modified_date = %s WHERE pet_id = %s",
        (datetime.now(), pet_id),
        commit=True,
    )
    return {"message": "Pet deleted successfully"}


# -------- PLACE ORDER --------
@app.post("/order")
async def place_order(
    user_id: int = Form(...),
    delivery_address: str = Form(...),
    payment_method: str = Form(...),
    special_notes: Optional[str] = Form(""),
    product_id: List[int] = Form(...),
    pet_id: List[int] = Form(...),
    quantity: List[int] = Form(...),
):
    if not (len(product_id) == len(pet_id) == len(quantity)):
        return JSONResponse(
            content={"error": "Mismatch in item array lengths"}, status_code=400
        )

    total_amount = 0
    order_date = datetime.now()
    created_date = modified_date = order_date

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Insert into orders table
        cursor.execute(
            """INSERT INTO orders 
            (user_id, order_date, status, total_amount, delivery_address, 
            delivery_date, payment_method, loyalty_points_earned, applied_discount, 
            special_notes, created_date, modified_date, is_delete) 
            VALUES (%s, %s, %s, %s, %s, NULL, %s, 0, 0, %s, %s, %s, 0)""",
            (
                user_id,
                order_date,
                "Pending",
                0.0,
                delivery_address,
                payment_method,
                special_notes,
                created_date,
                modified_date,
            ),
        )

        order_id = cursor.lastrowid

        # Insert order items + schedule notification
        for i in range(len(product_id)):
            prod_id = product_id[i]
            p_id = pet_id[i]
            qty = quantity[i]

            # Check pet belongs to user
            cursor.execute(
                "SELECT 1 FROM pets WHERE pet_id = %s AND user_id = %s AND is_delete = 0",
                (p_id, user_id),
            )
            if not cursor.fetchone():
                conn.rollback()
                return JSONResponse(
                    content={
                        "error": f"Pet ID {p_id} does not belong to User ID {user_id}"
                    },
                    status_code=400,
                )

            # Get product price and duration
            cursor.execute(
                "SELECT price, product_duration_days FROM products WHERE product_id = %s AND is_delete = 0",
                (prod_id,),
            )
            result = cursor.fetchone()
            if not result:
                conn.rollback()
                return JSONResponse(
                    content={"error": f"Product ID {prod_id} not found"},
                    status_code=404,
                )

            unit_price = float(result[0])
            duration_days = int(result[1] or 30)  # default 30 if null
            subtotal = qty * unit_price
            total_amount += subtotal

            # Insert into order_items
            cursor.execute(
                """INSERT INTO order_items 
                (order_id, product_id, pet_id, quantity, unit_price, subtotal, 
                created_date, modified_date, is_delete)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 0)""",
                (
                    order_id,
                    prod_id,
                    p_id,
                    qty,
                    unit_price,
                    subtotal,
                    created_date,
                    modified_date,
                ),
            )

            # Schedule reorder notification
            # sent_at = order_date + timedelta(days=duration_days)
            sent_at = datetime.now()

            cursor.execute(
                """INSERT INTO notification_log 
                (user_id, product_id, channel, message_type, sent_at, status, created_date, modified_date, is_delete)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 0)""",
                (
                    user_id,
                    prod_id,
                    "Email",
                    "ReorderReminder",
                    sent_at,
                    "Pending",
                    created_date,
                    modified_date,
                ),
            )

            notification_id = cursor.lastrowid

            make_payload = {
                "notification_id": notification_id,
                "user_id": user_id,
                "product_id": prod_id,
                "sent_at": sent_at.strftime("%Y-%m-%d %H:%M:%S"),
            }
            try:
                requests.post(
                    "https://hook.eu2.make.com/4bjei9zy950s5g2d57ryxvvb3nb5nur3",
                    json=make_payload,
                )
            except Exception as e:
                pass

        # Update total amount in orders table
        cursor.execute(
            "UPDATE orders SET total_amount = %s WHERE order_id = %s",
            (total_amount, order_id),
        )

        conn.commit()
    except Exception as e:
        conn.rollback()
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        cursor.close()
        conn.close()

    return JSONResponse(
        content={"message": "Order placed successfully", "order_id": order_id},
        status_code=201,
    )


# -------- SEND REORDER REMINDER(amFpbmFpbg==) --------
@app.post("/send-reorder-reminder")
def send_reorder_reminder(
    user_id: int = Form(...),
    product_id: int = Form(...),
    notification_id: int = Form(...),
):
    user = fetch_one("SELECT email, name FROM users WHERE user_id = %s", (user_id,))
    if not user:
        return JSONResponse({"error": "User not found"}, status_code=404)

    product = fetch_one(
        "SELECT name, description, price, photo_url FROM products WHERE product_id = %s",
        (product_id,),
    )
    if not product:
        return JSONResponse({"error": "Product not found"}, status_code=404)

    subject = f"Reminder to Reorder: {product['name']}"
    html_body = f"""
<html>
  <body style="margin:0; padding:0; font-family: Arial, sans-serif; background-color:#f6f6f6;">
    <table align="center" cellpadding="0" cellspacing="0" width="100%" style="max-width:600px; background-color:#ffffff; margin-top:30px; border-radius:8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
      
      <!-- Header -->
      <tr>
        <td align="center" bgcolor="#f4f4f4" style="padding: 40px 0;">
          <table width="600" cellpadding="0" cellspacing="0" style="background-color: #ffffff; border-radius: 8px; font-family: Arial, sans-serif;">
            <tr>
              <td align="center" style="padding: 20px;">
                <table cellpadding="0" cellspacing="0" style="margin: auto;">
                  <tr>
                    <!-- Logo Box -->
                    <td style="padding-right: 15px;">
                      <div style="width: 48px; height: 48px; background: linear-gradient(to bottom right, #3B82F6, #9333EA, #EC4899); border-radius: 12px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                        <span style="color: #ffffff; font-size: 24px;">🐾</span>
                      </div>
                    </td>
                    <!-- Store Name & Tagline -->
                    <td>
                      <div style="text-align: left;">
                        <h1 style="margin: 0; font-size: 22px; color: #1F2937; font-weight: bold;">Pet Goods Plus</h1>
                        <p style="margin: 4px 0 0; font-size: 12px; color: #6B7280;">Premium Pet Products & More</p>
                      </div>
                    </td>
                  </tr>
                </table>
              </td>
            </tr>

            <!-- Divider -->
            <tr>
              <td style="padding: 0 30px;">
                <hr style="border: none; border-top: 1px solid #e0e0e0;" />
              </td>
            </tr>
          </table>
        </td>
      </tr>
      <!-- End of Header -->

      <!-- Body -->
      <tr>
        <td style="padding: 30px;">
          <h2 style="color: #333;">Hi {user['name']},</h2>
          <p style="font-size:16px; color: #555;">It's time to reorder your favorite product!</p>
          
          <!-- Product Card -->
          <table width="100%" cellpadding="0" cellspacing="0" style="margin-top: 20px; border:1px solid #eee; border-radius:8px;">
            <tr>
              <td width="40%" style="padding:10px;">
                <img src="{product['photo_url']}" alt="{product['name']}" width="100%" style="border-radius:6px;" />
              </td>
              <td width="60%" style="padding: 10px; vertical-align: top;">
                <h3 style="margin: 0; color: #333;">{product['name']}</h3>
              </td>
            </tr>
          </table>

          <!-- CTA -->
          <div style="text-align:center; margin-top: 30px;">
            <a href="https://yourstore.com/reorder/{product_id}" 
               style="background-color: #1a73e8; color: white; text-decoration: none; padding: 12px 25px; border-radius: 5px; font-weight: bold; display:inline-block;">
              Reorder Now
            </a><br>
            
          </div>

          <!-- Chat Assistance (Email Safe, Mobile Friendly) -->
<table align="center" cellpadding="0" cellspacing="0" style="margin-top:30px;">
  <tr>
    <td colspan="3" align="center" style="padding-bottom:12px;">
      <p style="font-size:14px; color:#555; margin:0;">
        Need quick help? Chat with us on:
      </p>
    </td>
  </tr>
  <tr>
    <!-- Messenger -->
    <td align="center" width="60" style="padding: 0 0px;">
      <a href="https://m.me/182580971608579?text=Hey" target="_blank" style="text-decoration:none;">
        <img src="https://img.icons8.com/ios-filled/50/0084FF/facebook-messenger.png" 
             alt="Messenger" width="32" height="32" style="display:block;" />
      </a>
    </td>

    <!-- WhatsApp -->
    <td align="center" width="60" style="padding: 0 0px;">
      <a href="https://wa.me/15557642620?text=Hey" target="_blank" style="text-decoration:none;">
        <img src="https://img.icons8.com/ios-filled/50/25D366/whatsapp.png" 
             alt="WhatsApp" width="32" height="32" style="display:block;" />
      </a>
    </td>

    <!-- Instagram -->
    <td align="center" width="60" style="padding: 0 0px;">
      <a href="https://ig.me/m/smart_store_narola?text=hey" target="_blank" style="text-decoration:none;">
        <img src="https://img.icons8.com/ios-filled/50/000000/instagram-new.png" 
             alt="Instagram" width="32" height="32" style="display:block;" />
      </a>
    </td>
  </tr>
</table>


          <p style="font-size: 14px; color: #999; margin-top: 30px;">
            Thank you for choosing Your Store!<br>
            Need help? <a href="mailto:bloomai.notify@gmail.com" style="color:#1a73e8;">Contact Support</a>
          </p>
        </td>
      </tr>

      <!-- Footer -->
      <tr>
        <td style="text-align: center; font-size:12px; color:#aaa; padding:20px 0; border-top: 1px solid #eee;">
          © 2025 Your Store Inc. · <a href="#" style="color:#aaa; text-decoration:none;">Unsubscribe</a>
        </td>
      </tr>
    </table>
  </body>
</html>
"""
    send_email(user["email"], subject, html_body)

    execute_query(
        "UPDATE notification_log SET status = %s, modified_date = NOW() WHERE notification_id = %s",
        ("sent", notification_id),
        commit=True,
    )
    return {"message": "Reminder email sent successfully"}


# -------- GET USER ORDERS --------
@app.get("/orders/{user_id}")
def get_user_orders(user_id: int):
    return fetch_all(
        "SELECT * FROM orders WHERE user_id = %s AND is_delete = 0", (user_id,)
    )


# -------- CANCEL ORDER --------
@app.delete("/order/{order_id}")
def cancel_order(order_id: int):
    now = datetime.now()
    execute_many(
        [
            (
                "UPDATE orders SET is_delete = 1, status = 'Cancelled', modified_date = %s WHERE order_id = %s",
                (now, order_id),
            ),
            (
                "UPDATE order_items SET is_delete = 1, modified_date = %s WHERE order_id = %s",
                (now, order_id),
            ),
        ],
        commit=True,
    )
    return {"message": "Order cancelled successfully"}


# --- CART APIs ---
@app.post("/cart/add")
def add_to_cart(
    user_id: int = Form(...),
    product_id: int = Form(...),
    pet_id: int = Form(None),
    quantity: int = Form(...),
):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Check for active cart
    cursor.execute(
        "SELECT * FROM cart WHERE user_id=%s AND status='active'", (user_id,)
    )
    cart = cursor.fetchone()
    if not cart:
        cursor.execute("INSERT INTO cart (user_id) VALUES (%s)", (user_id,))
        conn.commit()
        cart_id = cursor.lastrowid
    else:
        cart_id = cart["cart_id"]

    # Get product price
    cursor.execute("SELECT price FROM products WHERE product_id=%s", (product_id,))
    product = cursor.fetchone()
    if not product:
        return {"error": "Product not found"}

    unit_price = float(product["price"])
    subtotal = unit_price * quantity

    # Add to cart_item
    cursor.execute(
        """
        INSERT INTO cart_item (cart_id, product_id, pet_id, quantity, unit_price, subtotal)
        VALUES (%s, %s, %s, %s, %s, %s)
    """,
        (cart_id, product_id, pet_id, quantity, unit_price, subtotal),
    )
    conn.commit()

    return {"message": "Added to cart", "cart_id": cart_id}


@app.get("/cart/{user_id}")
def view_cart(user_id: int):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Fetch active cart
    cursor.execute(
        "SELECT * FROM cart WHERE user_id=%s AND status='active'", (user_id,)
    )
    cart = cursor.fetchone()
    if not cart:
        return {"message": "Cart empty", "items": [], "grand_total": 0}

    # Fetch cart items with product details
    cursor.execute(
        """
        SELECT ci.cart_item_id, ci.product_id, p.name, ci.quantity, ci.unit_price, ci.subtotal
        FROM cart_item ci
        JOIN products p ON ci.product_id = p.product_id
        WHERE ci.cart_id=%s
    """,
        (cart["cart_id"],),
    )
    items = cursor.fetchall()

    # Calculate grand total
    grand_total = sum(item["subtotal"] for item in items)

    return {
        "cart_id": cart["cart_id"],
        "status": cart["status"],
        "items": items,
        "grand_total": grand_total,
    }


@app.put("/cart/update")
def update_cart_item(cart_item_id: int = Form(...), quantity: int = Form(...)):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Fetch cart item with cart status
    cursor.execute(
        """
        SELECT ci.unit_price, c.cart_id, c.status 
        FROM cart_item ci
        JOIN cart c ON ci.cart_id = c.cart_id
        WHERE ci.cart_item_id=%s
    """,
        (cart_item_id,),
    )
    item = cursor.fetchone()
    if not item:
        return {"error": "Cart item not found"}

    # Check cart status
    if item["status"] != "active":
        return {"message": "Cart is not active. Cannot update items."}

    # If quantity = 0 → remove item
    if quantity == 0:
        cursor.execute("DELETE FROM cart_item WHERE cart_item_id=%s", (cart_item_id,))
        conn.commit()
        return {"message": "Cart item removed"}

    # Else update quantity and subtotal
    new_subtotal = float(item["unit_price"]) * quantity
    cursor.execute(
        "UPDATE cart_item SET quantity=%s, subtotal=%s WHERE cart_item_id=%s",
        (quantity, new_subtotal, cart_item_id),
    )
    conn.commit()

    return {
        "message": "Cart item updated",
        "new_quantity": quantity,
        "new_subtotal": new_subtotal,
    }


@app.delete("/cart/remove")
def remove_cart_item(cart_item_id: int = Form(...)):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM cart_item WHERE cart_item_id=%s", (cart_item_id,))
    conn.commit()
    return {"message": "Item removed from cart"}


from datetime import datetime, timedelta
import requests


@app.post("/cart/checkout")
def checkout_cart(
    user_id: int = Form(...),
    delivery_address: str = Form(...),
    payment_method: str = Form(...),
    special_notes: str = Form(None),  # optional field
):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # --- 1. Fetch active cart ---
    cursor.execute(
        "SELECT * FROM cart WHERE user_id=%s AND status='active'", (user_id,)
    )
    cart = cursor.fetchone()
    if not cart:
        return {"error": "No active cart"}

    # --- 2. Fetch cart items ---
    cursor.execute("SELECT * FROM cart_item WHERE cart_id=%s", (cart["cart_id"],))
    items = cursor.fetchall()
    if not items:
        return {"error": "Cart empty"}

    total_amount = sum([i["subtotal"] for i in items])

    # --- 3. Create order ---
    order_date = datetime.now()
    delivery_date = datetime.now().date()  # set today's date
    cursor.execute(
        """
        INSERT INTO orders 
        (user_id, order_date, status, total_amount, delivery_address, payment_method, delivery_date, special_notes, created_date, modified_date, is_delete) 
        VALUES (%s, %s, 'Pending', %s, %s, %s, %s, %s, NOW(), NOW(), 0)
        """,
        (
            user_id,
            order_date,
            total_amount,
            delivery_address,
            payment_method,
            delivery_date,
            special_notes,
        ),
    )
    conn.commit()
    order_id = cursor.lastrowid

    # --- 4. Copy items into order_items ---
    for i in items:
        cursor.execute(
            """
            INSERT INTO order_items (order_id, product_id, pet_id, quantity, unit_price, subtotal) 
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                order_id,
                i["product_id"],
                i["pet_id"],
                i["quantity"],
                i["unit_price"],
                i["subtotal"],
            ),
        )

        # --- 5. Fetch duration_days from products ---
        cursor.execute(
            "SELECT product_duration_days FROM products WHERE product_id=%s",
            (i["product_id"],),
        )
        product = cursor.fetchone()
        duration_days = product.get("product_duration_days", 0) if product else 0

        # --- 6. Calculate reorder reminder date ---
        sent_at = (
            order_date + timedelta(days=duration_days) if duration_days else order_date
        )

        # --- 7. Insert into notification_log ---
        cursor.execute(
            """
            INSERT INTO notification_log 
            (user_id, product_id, channel, message_type, sent_at, status, created_date, modified_date, is_delete)
            VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW(), 0)
            """,
            (
                user_id,
                i["product_id"],
                "ManyChat",
                "ReorderReminder",
                sent_at,
                "Pending",
            ),
        )
        conn.commit()

        notification_id = cursor.lastrowid

        # --- 8. Send notification payload to ManyChat via Make webhook ---
        make_payload = {
            "notification_id": notification_id,
            "user_id": user_id,
            "product_id": i["product_id"],
            # "sent_at": sent_at.strftime("%Y-%m-%d %H:%M:%S"),
            "sent_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            requests.post(
                "https://hook.eu2.make.com/4bjei9zy950s5g2d57ryxvvb3nb5nur3",
                json=make_payload,
                timeout=5,
            )
        except Exception as e:
            print("ManyChat webhook error:", str(e))
            pass

    conn.commit()

    # --- 9. Mark cart as checked_out ---
    cursor.execute(
        "UPDATE cart SET status='checked_out' WHERE cart_id=%s", (cart["cart_id"],)
    )
    conn.commit()
    cursor.execute("UPDATE orders SET status='paid' WHERE order_id=%s", (order_id,))
    conn.commit()
    return {
        "message": "Order placed successfully",
        "order_id": order_id,
        "delivery_date": str(delivery_date),
        "special_notes": special_notes,
    }


# -------- PET JOURNAL --------
@app.post("/user/pet-journals")
def get_user_pet_journals(
    user_id: int = Form(..., description="User ID"),
    pet_id: int = Form(None, description="Pet ID (optional)"),
):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        query = """
            SELECT 
                pj.journal_id,
                pj.pet_id,
                p.name AS pet_name,
                p.species,
                pj.entry_date,
                pj.entry_type,
                pj.details,
                pj.created_date,
                pj.modified_date
            FROM pet_journals pj
            INNER JOIN pets p ON pj.pet_id = p.pet_id
            WHERE p.user_id = %s AND pj.is_delete = 0 AND p.is_delete = 0
        """
        params = [user_id]

        if pet_id is not None:
            query += " AND pj.pet_id = %s"
            params.append(pet_id)

        query += " ORDER BY pj.entry_date DESC"

        cursor.execute(query, tuple(params))
        journals = cursor.fetchall()

        if not journals:
            return {
                "user_id": user_id,
                "pet_id": pet_id,
                "journals": [],
                "message": "No pet journals found for this filter",
            }

        return {"user_id": user_id, "pet_id": pet_id, "journals": journals}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if "cursor" in locals():
            cursor.close()
        if "conn" in locals():
            conn.close()


# -------- WhatsApp Integration --------
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import mysql.connector, ast, os
from langchain.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from sqlalchemy import create_engine
from langchain_community.utilities.sql_database import SQLDatabase
import os
import json
import requests
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv
from whatsapp_helper import (
    llm,
    MANYCHAT_API_KEY,
    username,
    password,
    host,
    port,
    database_name,
    agent_executor,
    conversation_history,
    pincode_requests,
    engine,
    sql_prefix,
    get_user_profile,
    detect_if_greeting,
    is_delivery_concern,
    get_join_hint,
)

load_dotenv(override=True)


conversation_history = {}

# Load metadata
with open("clyro_metadata.json", "r") as f:
    metadata = json.load(f)


join_hint = get_join_hint(metadata)


db = SQLDatabase(engine)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

import json
import ast
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


def save_conversation(
    platform,
    user_email=None,
    session_id=None,
    user_message=None,
    ai_response=None,
    username=None,
    phone=None,
    user_id=None,
):
    """
    Save a conversation exchange (user message + AI response)
    in a single row inside clyro_agent_chat_logs.
    """
    if not user_message and not ai_response:
        return  # Nothing to save

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO clyro_agent_chat_logs 
            (user_email, message, ai_response, timestamp, user_id, session_id, platform, created_at, username, phone)
            VALUES (%s, %s, %s, NOW(), %s, %s, %s, NOW(), %s, %s)
        """,
            (
                user_email,
                user_message,
                ai_response,
                user_id,
                session_id,
                platform,
                username,
                phone,
            ),
        )

        conn.commit()
        print(f"[DB] Conversation saved for platform={platform}, session={session_id}")

    except Exception as e:
        print(f"[DB ERROR] {e}")
    finally:
        cursor.close()
        conn.close()


# WhatsApp webhook
@app.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    data = await request.json()

    user_id = data.get("id")
    message = data.get("message")
    user_info = data.get("user_details", {})

    user_name = user_info.get("first_name") or user_info.get("name")
    user_phone = user_info.get("whatsapp_phone") or user_info.get("phone")

    if not user_id or not message:
        return JSONResponse({"error": "Invalid payload"}), 400

    try:
        session_id = f"whatsapp-{user_phone or user_id}"

        # Fetch user profile
        profile = get_user_profile(user_phone) if user_phone else None
        extra_intro = ""
        # print("USER PROFILE:", profile)
        if not profile:
            profile = None
        # Check if greeting
        greeting_response = detect_if_greeting(message, user_name)
        if greeting_response != "NOT_A_GREETING":
            reply_text = greeting_response
            save_conversation(
                platform="whatsapp",
                user_email=profile["user"]["email"] if profile else None,
                session_id=session_id,
                user_message=message,
                ai_response=reply_text,
                username=user_name,
                phone=user_phone,
                user_id=user_id,
            )
            # Save to memory
            conversation_history.setdefault(session_id, []).append(
                (message, reply_text)
            )
            headers = {
                "Authorization": f"Bearer {MANYCHAT_API_KEY}",
                "Content-Type": "application/json",
            }

            mc_payload = {
                "subscriber_id": user_id,
                "data": {
                    "version": "v2",
                    "content": {
                        "type": "whatsapp",
                        "messages": [{"type": "text", "text": reply_text}],
                    },
                },
            }

            r = requests.post(
                "https://api.manychat.com/fb/sending/sendContent",
                json=mc_payload,
                headers=headers,
            )

            return JSONResponse({"status": "success", "reply": reply_text})

        if is_delivery_concern(message):
            session_pin = pincode_requests.get(session_id, {})
            if not session_pin.get("pincode"):
                # Ask for pincode
                pincode_requests[session_id] = {"asked": True}
                pincode_requests[session_id] = {"asked": True, "concern": message}
                ask_pin_reply = "Please share your pincode so I can check delivery availability for your location."
                save_conversation(
                    platform="whatsapp",
                    user_email=profile["user"]["email"] if profile else None,
                    session_id=session_id,
                    user_message=message,
                    ai_response=ask_pin_reply,
                    username=user_name,
                    phone=user_phone,
                    user_id=user_id,
                )
                conversation_history.setdefault(session_id, []).append(
                    (message, ask_pin_reply)
                )

                headers = {
                    "Authorization": f"Bearer {MANYCHAT_API_KEY}",
                    "Content-Type": "application/json",
                }

                mc_payload = {
                    "subscriber_id": user_id,
                    "data": {
                        "version": "v2",
                        "content": {
                            "type": "whatsapp",
                            "messages": [{"type": "text", "text": ask_pin_reply}],
                        },
                    },
                }

                r = requests.post(
                    "https://api.manychat.com/fb/sending/sendContent",
                    json=mc_payload,
                    headers=headers,
                )

                return JSONResponse({"status": "awaiting_pincode"})

        # Pincode response handler
        if pincode_requests.get(session_id, {}).get("asked"):
            extraction_prompt = f"""
            Extract only the pincode (5 or 6 digit number) from this user message:
            "{message}"

            If no valid pincode is found, return "NONE".
            """

            extracted = llm.invoke(extraction_prompt).content.strip()

            if extracted.isdigit() and len(extracted) in [5, 6]:
                pincode = extracted
                pincode_requests[session_id]["pincode"] = pincode
                user_concern = pincode_requests.get(session_id, {}).get(
                    "concern", message
                )
                if not profile and user_phone:
                    profile = get_user_profile(user_phone)

                if profile:
                    user_tuple = profile["user"]
                    payload = {
                        "user_concern": user_concern,
                        "user_pincode": message,
                        "phone": user_phone,
                        "email": profile["user"]["email"],
                        "name": profile["user"]["name"],
                    }
                    make_webhook_url = os.getenv(
                        "make_webhook_url"
                    )  # Replace with your URL

                    try:
                        response = requests.post(make_webhook_url, json=payload)

                    except Exception as e:
                        pass

                else:
                    pass
                # your existing profile / webhook / reply logic
                delivery_reply = f"Thanks for sharing your pincode ({pincode}). Our team ensures timely delivery in most areas. You can proceed confidently with your order. 😊"
                save_conversation(
                    platform="whatsapp",
                    user_email=profile["user"]["email"] if profile else None,
                    session_id=session_id,
                    user_message=message,
                    ai_response=delivery_reply,
                    username=user_name,
                    phone=user_phone,
                    user_id=user_id,
                )
                conversation_history.setdefault(session_id, []).append(
                    (message, delivery_reply)
                )

                headers = {
                    "Authorization": f"Bearer {MANYCHAT_API_KEY}",
                    "Content-Type": "application/json",
                }
                mc_payload = {
                    "subscriber_id": user_id,
                    "data": {
                        "version": "v2",
                        "content": {
                            "type": "whatsapp",
                            "messages": [{"type": "text", "text": delivery_reply}],
                        },
                    },
                }
                r = requests.post(
                    "https://api.manychat.com/fb/sending/sendContent",
                    json=mc_payload,
                    headers=headers,
                )

                return JSONResponse(
                    {"status": "pincode_received", "reply": delivery_reply}
                )

        # if pincode_requests.get(session_id, {}).get("asked") and message.strip().isdigit() and len(message.strip()) in [5, 6]:
        #     pincode = message.strip()
        #     pincode_requests[session_id]["pincode"] = pincode
        #     user_concern = pincode_requests.get(session_id, {}).get("concern", message)
        #     if not profile and user_phone:
        #         profile = get_user_profile(
        #             user_phone
        #         )

        #     if profile:
        #         user_tuple = profile["user"]
        #         payload = {
        #             "user_concern": user_concern,
        #             "user_pincode": message,
        #             "phone": user_phone,
        #             "email": profile["user"]["email"],
        #             "name": profile["user"]["name"],
        #         }
        #         make_webhook_url = os.getenv(
        #             "make_webhook_url"
        #         )  # Replace with your URL

        #         try:
        #             response = requests.post(make_webhook_url, json=payload)

        #         except Exception as e:
        #             pass

        #     else:
        #         pass

        #     delivery_reply = f"Thanks for sharing your pincode ({pincode}). Our team ensures timely delivery in most areas. You can proceed confidently with your order. 😊"

        #     conversation_history.setdefault(session_id, []).append(
        #         (message, delivery_reply)
        #     )

        #     headers = {
        #         "Authorization": f"Bearer {MANYCHAT_API_KEY}",
        #         "Content-Type": "application/json",
        #     }

        #     mc_payload = {
        #         "subscriber_id": user_id,
        #         "data": {
        #             "version": "v2",
        #             "content": {
        #                 "type": "whatsapp",
        #                 "messages": [{"type": "text", "text": delivery_reply}],
        #             },
        #         },
        #     }

        #     r = requests.post(
        #         "https://api.manychat.com/fb/sending/sendContent",
        #         json=mc_payload,
        #         headers=headers,
        #     )
        #     # print("ManyChat response:", r.status_code, r.text)

        #     return JSONResponse({"status": "pincode_received", "reply": delivery_reply})

        history = conversation_history.get(session_id, [])
        formatted_history = "\n".join(
            [f"User: {q}\nAssistant: {a}" for q, a in history[-5:]]
        )

        rewrite_prompt = f"""
        You are an assistant that decides whether a user message should be rewritten into a **shopping/product search query** 
        OR handled as a **service/support concern**.

        Here is the previous conversation:
        {formatted_history}

        👤 User details with user's pet details:
        {profile}

        The user’s current message is:
        "{message}"

        ➡️ Instructions:
        1. If the message is about **shopping or product needs** (food, supplements, treats, medicine, toys, accessories, etc.), 
        rewrite it into a precise shopping query including:
        - Pet details (species, breed, age, allergies).
        - Explicit product category.
        - Any restrictions (e.g., avoid chicken if allergic).
        Return ONLY the rewritten product query.

        2. If the message is about **delivery, cancellation, refund, damaged product, or order issues**, 
        DO NOT rewrite it into a product query.  
        Instead, return:  
        "SUPPORT: <summarized user concern>"

        3. If it is a general greeting, return:  
        "GREETING"

        ⚠️ Important: Only output the rewritten query, "SUPPORT: ..." or "GREETING". 
        No explanations, no extra text.
        """

        rewritten_question = llm.invoke(rewrite_prompt).content

        reply_text = agent_executor.run(rewritten_question)
        # 🌍 Translate if needed
        translation_prompt = f"""
        You are a helpful assistant.

        The user's original message was:
        "{message}"

        The assistant's reply is:
        "{reply_text}"

        If the user's message is written in clear English, return the assistant's reply exactly as-is without any translation.  
        If the user's message is written in a non-English language, translate the assistant's reply into that exact same language, keeping the tone friendly and conversational.  

        Return ONLY the final reply (translated or original) — no explanations, no extra text.
        """

        reply_text = llm.invoke(translation_prompt).content.strip()
        save_conversation(
            platform="whatsapp",
            user_email=profile["user"]["email"] if profile else None,
            session_id=session_id,
            user_message=message,
            ai_response=reply_text,
            username=user_name,
            phone=user_phone,
            user_id=user_id,
        )
        # Save conversation
        history.append((message, reply_text))
        conversation_history[session_id] = history

        headers = {
            "Authorization": f"Bearer {MANYCHAT_API_KEY}",
            "Content-Type": "application/json",
        }

        mc_payload = {
            "subscriber_id": user_id,
            "data": {
                "version": "v2",
                "content": {
                    "type": "whatsapp",
                    "messages": [{"type": "text", "text": reply_text}],
                },
            },
        }

        r = requests.post(
            "https://api.manychat.com/fb/sending/sendContent",
            json=mc_payload,
            headers=headers,
        )

        return JSONResponse({"status": "success", "reply": reply_text})

    except Exception as e:
        return JSONResponse({"error": str(e)}), 500


# --------------- whatsapp bot end -------------------

# --------------- instagram bot start -------------------
from instagram_helper import (
    get_user_profile_by_ig,
    send_to_manychat,
    conversation_history,
    pincode_requests,
)


@app.post("/instagram")
async def instagram_webhook(request: Request):
    data = await request.json()

    user_id = data.get("id")
    message = data.get("message")
    user_info = data.get("user_details", {})

    user_name = user_info.get("first_name") or user_info.get("name")
    ig_username = user_info.get("ig_username")  # Use IG handle
    conn = get_db_connection()
    cursor = conn.cursor()
    user_phone = None
    if ig_username:
        query = (
            "SELECT phone, email FROM users WHERE ig_username = %s AND is_delete = 0"
        )
        cursor.execute(query, (ig_username,))
        result = cursor.fetchone()
        if result:
            user_phone = result[0]
            user_email = result[1]
        else:
            user_phone = None
            user_email = None
    else:
        user_phone = None
        user_email = None
    # print(f"[WHATSAPP WEBHOOK] From {user_id}: {message} (User: {user_name}, Phone: {user_phone})")

    if not user_id or not message:
        return JSONResponse({"error": "Invalid payload"}), 400

    try:
        session_id = f"instagram-{ig_username or user_id}"

        # Fetch user profile
        profile = get_user_profile(user_phone) if user_phone else None
        extra_intro = ""
        # print("USER PROFILE:", profile)
        if not profile:
            profile = None

        # Check if greeting
        greeting_response = detect_if_greeting(message, user_name)
        if greeting_response != "NOT_A_GREETING":
            reply_text = greeting_response
            # Save to memory
            save_conversation(
                platform="instagram",
                user_email=user_email,
                session_id=session_id,
                user_message=message,
                ai_response=reply_text,
                username=ig_username or user_name,
                phone=user_phone,
                user_id=user_id,
            )
            conversation_history.setdefault(session_id, []).append(
                (message, reply_text)
            )

            headers = {
                "Authorization": f"Bearer {MANYCHAT_API_KEY}",
                "Content-Type": "application/json",
            }
            mc_payload = {
                "subscriber_id": user_id,
                "data": {
                    "version": "v2",
                    "content": {
                        "type": "instagram",
                        "messages": [{"type": "text", "text": reply_text}],
                    },
                },
            }
            r = requests.post(
                "https://api.manychat.com/fb/sending/sendContent",
                json=mc_payload,
                headers=headers,
            )

            return JSONResponse({"status": "success", "reply": reply_text})

        if is_delivery_concern(message):
            session_pin = pincode_requests.get(session_id, {})
            if not session_pin.get("pincode"):
                # Ask for pincode
                pincode_requests[session_id] = {"asked": True}
                pincode_requests[session_id] = {"asked": True, "concern": message}
                ask_pin_reply = "Please share your pincode so I can check delivery availability for your location."
                save_conversation(
                    platform="instagram",
                    user_email=user_email,
                    session_id=session_id,
                    user_message=message,
                    ai_response=ask_pin_reply,
                    username=ig_username or user_name,
                    phone=user_phone,
                    user_id=user_id,
                )
                conversation_history.setdefault(session_id, []).append(
                    (message, ask_pin_reply)
                )

                headers = {
                    "Authorization": f"Bearer {MANYCHAT_API_KEY}",
                    "Content-Type": "application/json",
                }

                mc_payload = {
                    "subscriber_id": user_id,
                    "data": {
                        "version": "v2",
                        "content": {
                            "type": "instagram",
                            "messages": [{"type": "text", "text": ask_pin_reply}],
                        },
                    },
                }

                r = requests.post(
                    "https://api.manychat.com/fb/sending/sendContent",
                    json=mc_payload,
                    headers=headers,
                )

                return JSONResponse({"status": "awaiting_pincode"})

        # Pincode response handler
        # if pincode_requests.get(session_id, {}).get("asked") and message.strip().isdigit() and len(message.strip()) in [5, 6]:

        #     pincode = message.strip()
        #     pincode_requests[session_id]["pincode"] = pincode
        #     user_concern = pincode_requests.get(session_id, {}).get("concern", message)
        #     if not profile and user_phone:
        #         profile = get_user_profile(
        #             user_phone
        #         )

        #     if profile:
        #         user_tuple = profile["user"]
        #         payload = {
        #             "user_concern": user_concern,
        #             "user_pincode": message,
        #             "phone": user_phone,
        #             "email": profile["user"]["email"],
        #             "name": profile["user"]["name"],
        #         }
        #         make_webhook_url = os.getenv(
        #             "make_webhook_url"
        #         )  # Replace with your URL

        #         try:
        #             response = requests.post(make_webhook_url, json=payload)
        #         except Exception as e:
        #             pass
        #     else:
        #         pass

        #     delivery_reply = f"Thanks for sharing your pincode ({pincode}). Our team ensures timely delivery in most areas. You can proceed confidently with your order. 😊"

        #     conversation_history.setdefault(session_id, []).append(
        #         (message, delivery_reply)
        #     )

        #     headers = {
        #         "Authorization": f"Bearer {MANYCHAT_API_KEY}",
        #         "Content-Type": "application/json",
        #     }

        #     mc_payload = {
        #         "subscriber_id": user_id,
        #         "data": {
        #             "version": "v2",
        #             "content": {
        #                 "type": "instagram",
        #                 "messages": [{"type": "text", "text": delivery_reply}],
        #             },
        #         },
        #     }

        #     r = requests.post(
        #         "https://api.manychat.com/fb/sending/sendContent",
        #         json=mc_payload,
        #         headers=headers,
        #     )

        #     return JSONResponse({"status": "pincode_received", "reply": delivery_reply})
        if pincode_requests.get(session_id, {}).get("asked"):
            extraction_prompt = f"""
            Extract only the pincode (5 or 6 digit number) from this user message:
            "{message}"

            If no valid pincode is found, return "NONE".
            """

            extracted = llm.invoke(extraction_prompt).content.strip()

            if extracted.isdigit() and len(extracted) in [5, 6]:
                pincode = extracted
                pincode_requests[session_id]["pincode"] = pincode
                user_concern = pincode_requests.get(session_id, {}).get(
                    "concern", message
                )
                if not profile and user_phone:
                    profile = get_user_profile(user_phone)

                if profile:
                    user_tuple = profile["user"]
                    payload = {
                        "user_concern": user_concern,
                        "user_pincode": message,
                        "phone": user_phone,
                        "email": profile["user"]["email"],
                        "name": profile["user"]["name"],
                    }
                    make_webhook_url = os.getenv(
                        "make_webhook_url"
                    )  # Replace with your URL

                    try:
                        response = requests.post(make_webhook_url, json=payload)

                    except Exception as e:
                        pass

                else:
                    pass
                # your existing profile / webhook / reply logic
                delivery_reply = f"Thanks for sharing your pincode ({pincode}). Our team ensures timely delivery in most areas. You can proceed confidently with your order. 😊"
                save_conversation(
                    platform="instagram",
                    user_email=user_email,
                    session_id=session_id,
                    user_message=message,
                    ai_response=delivery_reply,
                    username=ig_username or user_name,
                    phone=user_phone,
                    user_id=user_id,
                )
                conversation_history.setdefault(session_id, []).append(
                    (message, delivery_reply)
                )

                headers = {
                    "Authorization": f"Bearer {MANYCHAT_API_KEY}",
                    "Content-Type": "application/json",
                }
                mc_payload = {
                    "subscriber_id": user_id,
                    "data": {
                        "version": "v2",
                        "content": {
                            "type": "instagram",
                            "messages": [{"type": "text", "text": delivery_reply}],
                        },
                    },
                }
                r = requests.post(
                    "https://api.manychat.com/fb/sending/sendContent",
                    json=mc_payload,
                    headers=headers,
                )

                return JSONResponse(
                    {"status": "pincode_received", "reply": delivery_reply}
                )

        history = conversation_history.get(session_id, [])
        formatted_history = "\n".join(
            [f"User: {q}\nAssistant: {a}" for q, a in history[-5:]]
        )

        rewrite_prompt = f"""
        You are an assistant that decides whether a user message should be rewritten into a **shopping/product search query** 
        OR handled as a **service/support concern**.

        Here is the previous conversation:
        {formatted_history}

        👤 User details with user's pet details:
        {profile}

        The user’s current message is:
        "{message}"

        ➡️ Instructions:
        1. If the message is about **shopping or product needs** (food, supplements, treats, medicine, toys, accessories, etc.), 
        rewrite it into a precise shopping query including:
        - Pet details (species, breed, age, allergies).
        - Explicit product category.
        - Any restrictions (e.g., avoid chicken if allergic).
        Return ONLY the rewritten product query.

        2. If the message is about **delivery, cancellation, refund, damaged product, or order issues**, 
        DO NOT rewrite it into a product query.  
        Instead, return:  
        "SUPPORT: <summarized user concern>"

        3. If it is a general greeting, return:  
        "GREETING"

        ⚠️ Important: Only output the rewritten query, "SUPPORT: ..." or "GREETING". 
        No explanations, no extra text.
        """

        rewritten_question = llm.invoke(rewrite_prompt).content
        # print("Rewritten:", rewritten_question)

        reply_text = agent_executor.run(rewritten_question)

        # 🌍 Translate if needed
        translation_prompt = f"""
        You are a helpful assistant.

        The user's original message was:
        "{message}"

        The assistant's reply is:
        "{reply_text}"

        If the user's message is written in clear English, return the assistant's reply exactly as-is without any translation.  
        If the user's message is written in a non-English language, translate the assistant's reply into that exact same language, keeping the tone friendly and conversational.  

        Return ONLY the final reply (translated or original) — no explanations, no extra text.
        """

        reply_text = llm.invoke(translation_prompt).content.strip()
        save_conversation(
            platform="instagram",
            user_email=user_email,
            session_id=session_id,
            user_message=message,
            ai_response=reply_text,
            username=ig_username or user_name,
            phone=user_phone,
            user_id=user_id,
        )
        # Save conversation
        history.append((message, reply_text))
        conversation_history[session_id] = history

        headers = {
            "Authorization": f"Bearer {MANYCHAT_API_KEY}",
            "Content-Type": "application/json",
        }

        mc_payload = {
            "subscriber_id": user_id,
            "data": {
                "version": "v2",
                "content": {
                    "type": "instagram",
                    "messages": [{"type": "text", "text": reply_text}],
                },
            },
        }

        r = requests.post(
            "https://api.manychat.com/fb/sending/sendContent",
            json=mc_payload,
            headers=headers,
        )

        return JSONResponse({"status": "success", "reply": reply_text})

    except Exception as e:
        return JSONResponse({"error": str(e)}), 500


# ----------------- End instagram bot -----------------

#  --------------- Messenger bot start -------------------

from messenger_helpers import (
    messenger_pincode_requests,
    conversation_history,
    create_support_chain,
    send_messenger_response,
    ensure_language_consistency,
    messenger_prompt,
)


def create_support_chain():
    return ChatPromptTemplate.from_template(messenger_prompt) | llm


@app.post("/messenger")
async def messenger_webhook(request: Request):
    data = await request.json()
    message = data.get("message", "").strip()
    user_id = data.get("id")
    session_id = f"messenger-{user_id}"
    user_name = data.get("user_details", {}).get("first_name", "there")

    if not message:
        return JSONResponse({"error": "Empty message"}), 400

    try:
        # Retrieve conversation history

        history = conversation_history.get(session_id, [])
        formatted_history = "\n".join(
            [f"User: {q}\nAssistant: {a}" for q, a in history[-3:]]
        )

        # Handle greetings
        greeting_response = detect_if_greeting(message, user_name)
        if greeting_response != "NOT_A_GREETING":
            save_conversation(
                platform="messenger",
                user_email=None,
                session_id=session_id,
                user_message=message,
                ai_response=greeting_response,
                username=user_name,
                phone=None,
                user_id=user_id,
            )
            conversation_history.setdefault(session_id, []).append(
                (message, greeting_response)
            )
            return send_messenger_response(
                user_id, greeting_response, session_id, message, user_name
            )

        # Pincode handling logic
        pincode_data = messenger_pincode_requests.get(session_id, {})

        # Check if awaiting pincode
        if (
            pincode_data.get("asked")
            and message.strip().isdigit()
            and len(message.strip()) in [5, 6]
        ):
            pincode = message.strip()
            # Log pincode (implement your logging here)

            # Build reassurance message
            reply = (
                f"Thanks for sharing your pincode ({pincode})! "
                "We deliver to your area within 3-5 business days. "
                "Rest assured your order will arrive safely with our COD service."
            )
            save_conversation(
                platform="messenger",
                user_email=None,
                session_id=session_id,
                user_message=message,
                ai_response=reply,
                username=user_name,
                phone=None,
                user_id=user_id,
            )
            # Clear pincode state
            messenger_pincode_requests.pop(session_id, None)
            return send_messenger_response(
                user_id, reply, session_id, message, user_name
            )

        # Detect delivery concerns
        if is_delivery_concern(message) and not pincode_data.get("asked"):
            # Set pincode request state
            messenger_pincode_requests[session_id] = {"asked": True, "concern": message}
            ask_pin_reply = (
                "To check delivery for your area, " "please share your 6-digit pincode."
            )
            save_conversation(
                platform="messenger",
                user_email=None,
                session_id=session_id,
                user_message=message,
                ai_response=ask_pin_reply,
                username=user_name,
                phone=None,
                user_id=user_id,
            )
            return send_messenger_response(
                user_id, ask_pin_reply, session_id, message, user_name
            )

        # Generate support response
        support_chain = create_support_chain()
        response = support_chain.invoke(
            {"history": formatted_history, "input": message}
        ).content
        print("response:", response)
        # Handle language consistency
        if not pincode_data.get("asked"):  # Don't translate pincode requests
            response = ensure_language_consistency(message, response, llm)
        save_conversation(
            platform="messenger",
            user_email=None,
            session_id=session_id,
            user_message=message,
            ai_response=response,
            username=user_name,
            phone=None,
            user_id=user_id,
        )
        return send_messenger_response(
            user_id, response, session_id, message, user_name
        )

    except Exception as e:
        error_msg = (
            f"Sorry {user_name}, I'm having trouble. Please contact support@clyro.com"
        )
        save_conversation(
            platform="messenger",
            user_email=None,
            session_id=session_id,
            user_message=message,
            ai_response=error_msg,
            username=user_name,
            phone=None,
            user_id=user_id,
        )
        return send_messenger_response(
            user_id, error_msg, session_id, message, user_name
        )


def send_messenger_response(user_id, text, session_id, user_message, user_name):
    # Send via ManyChat
    headers = {
        "Authorization": f"Bearer {MANYCHAT_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "subscriber_id": user_id,
        "data": {
            "version": "v2",
            "content": {
                "type": "facebook",  # Messenger specific
                "messages": [{"type": "text", "text": text}],
            },
        },
    }

    requests.post(
        "https://api.manychat.com/fb/sending/sendContent",
        json=payload,
        headers=headers,
    )

    # Update conversation history
    history = conversation_history.get(session_id, [])
    history.append((user_message, text))
    conversation_history[session_id] = history

    return JSONResponse({"status": "success"})


# ----------------- End Messenger bot -----------------
@app.post("/send-mail")
async def send_mail_endpoint(request: Request):
    # 1. Extract values
    body = await request.json()
    user_email = body.get("email")
    user_question = body.get("question", "")

    print(f"📩 Email received: {user_email}")
    print(f"🤔 User asked for: {user_question}")

    # 2. Generate product recommendation with LLM (mock example)
    recommendation = f"""
        You are an assistant that rewrites user messages into **precise product search queries** for SQL query processing.

        The user’s current message is:
        "{user_question}"

        ➡️ Rewrite the user’s message into a full shopping/product-related query that includes:
        1. The correct pet name and its details (species, breed, age, allergies).
        2. Explicit product category (food, supplements, treats, medicine, accessories, etc.).
        3. Any relevant filters like breed/species restrictions (e.g., avoid chicken if allergic).
        
        ⚠️ Only return the rewritten query. Do NOT ask the user for pet details again — always use the database info.
        ### Rules:
            1. Always return output in strict JSON format only.
            2. JSON structure:
            {{
            "query": "<refined query>",
            "filters": {{
                "pet": "<species/breed/age/allergies>",
                "category": "<product category like food, supplements, accessories>",
                "restrictions": "<any exclusions like avoid chicken>"
            }}
            }}
        """
    rewritten_question = llm.invoke(recommendation).content
    print("🔄 Rewritten question:", rewritten_question)
    recommendations = agent_executor.run(rewritten_question)
    print("🛍️ Recommendations generated:", recommendations)
    recommendations_prompt = f"""
        Format the product recommendations strictly as JSON.

        Expected schema:
        [
        {{
            "title": "string",
            "description": "string",
            "price": "string",
            "image_url": "string",
            "product_url": "string"
        }}
        ]

        Here are the raw recommendations:
        {recommendations}
        """

    recommendations_json = llm.invoke(recommendations_prompt).content
    # 3. Send email
    import json

    products = json.loads(recommendations_json)
    subject = "Your Product Recommendations"
    html_items = ""
    for p in products:
        html_items += f"""
        <div style="border:1px solid #ddd; padding:15px; margin:10px 0; border-radius:10px;">
            <img src="{p['image_url']}" alt="{p['title']}" style="width:150px; border-radius:8px;"><br>
            <h3 style="margin:5px 0;">{p['title']}</h3>
            <p>{p['description']}</p>
            <p><b>Price:</b> {p['price']}</p>
            <a href="{p['product_url']}" 
            style="display:inline-block; padding:10px 15px; background:#007BFF; color:#fff; 
                    text-decoration:none; border-radius:5px;">View Product</a>
        </div>
        """

    html_body = f"""
    <h2>✨ Here are your personalized product recommendations ✨</h2>
    {html_items}
    <p style="margin-top:20px;">🐾 Thank you for shopping with us! 🐾</p>
    """
    print("html_body:", html_body)
    send_email(user_email, subject, html_body)
    print(f"✅ Recommendation email sent to {user_email}")
    return {"status": "success", "message": "Email sent successfully"}
