import { useState, useRef, useEffect, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { buildApiUrl } from '../lib/api'
import { useDraftStore } from '../stores/draftStore'
import { DRAFT_YEAR } from '../lib/config'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

const SUGGESTED_QUESTIONS = [
  "Which teams need a center most?",
  "How does Kyle Dubas typically draft?",
  "Who are the top defensive prospects this year?",
  "What's Tampa Bay's drafting tendency?",
  "Which prospects are being drafted below their CSS rank?",
]

function MessageBubble({ msg, index }: { msg: Message; index: number }) {
  const isUser = msg.role === 'user'
  return (
    <motion.div
      key={index}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-3`}
    >
      {!isUser && (
        <div className="w-7 h-7 rounded-full bg-accent-blue/20 border border-accent-blue/40 flex items-center justify-center flex-shrink-0 mr-2 mt-0.5">
          <span className="text-xs text-accent-blue font-bold">S</span>
        </div>
      )}
      <div
        className={`max-w-[80%] rounded-xl px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap ${
          isUser
            ? 'bg-accent-blue text-white rounded-br-sm'
            : 'bg-bg-card border border-border-subtle text-text-primary rounded-bl-sm'
        }`}
      >
        {msg.content}
      </div>
    </motion.div>
  )
}

export default function ScoutPage() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [toolStatus, setToolStatus] = useState<string | null>(null)
  const [sessionId] = useState(() => crypto.randomUUID())
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const { lotteryResult, draftResult } = useDraftStore()

  // Build draft context when an active simulation is in progress.
  // Passed with every message so the Scout can give pick-specific advice.
  const draftContext = useMemo(() => {
    if (!lotteryResult.length) return undefined
    const pickingTeam = lotteryResult[draftResult.length]
    return {
      current_pick: draftResult.length + 1,
      picking_team: pickingTeam?.team_name,
      picks_made: draftResult.length,
      recent_picks: draftResult.slice(-3).map(p => ({
        pick: p.pick,
        team: p.abbreviation,
        prospect: p.prospect_name,
        position: p.position,
      })),
    }
  }, [lotteryResult, draftResult])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  async function sendMessage(text: string) {
    const trimmed = text.trim()
    if (!trimmed || loading) return

    setMessages(prev => [...prev, { role: 'user', content: trimmed }])
    setInput('')
    setLoading(true)

    // Add empty assistant message we'll fill in as tokens arrive
    setMessages(prev => [...prev, { role: 'assistant', content: '' }])

    try {
      const res = await fetch(buildApiUrl('/api/agent/chat/stream'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
        message: trimmed,
        session_id: sessionId,
        ...(draftContext ? { draft_context: draftContext } : {}),
      }),
      })
      if (!res.ok || !res.body) throw new Error('Stream unavailable')

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const payload = JSON.parse(line.slice(6))
            if (payload.status) {
              setToolStatus(payload.status)
            }
            if (payload.token) {
              setToolStatus(null)
              setMessages(prev => {
                const next = [...prev]
                next[next.length - 1] = {
                  role: 'assistant',
                  content: next[next.length - 1].content + payload.token,
                }
                return next
              })
            }
            if (payload.done || payload.error) {
              setToolStatus(null)
              setLoading(false)
            }
          } catch { /* skip */ }
        }
      }
    } catch {
      setToolStatus(null)
      setMessages(prev => {
        const next = [...prev]
        next[next.length - 1] = { role: 'assistant', content: 'Sorry, I could not reach the server. Please try again.' }
        return next
      })
    } finally {
      setToolStatus(null)
      setLoading(false)
      inputRef.current?.focus()
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage(input)
    }
  }

  const isEmpty = messages.length === 0

  return (
    <main className="max-w-3xl mx-auto px-4 py-8 flex flex-col" style={{ height: 'calc(100vh - 3.5rem)' }}>
      {/* Header */}
      <div className="mb-4 flex-shrink-0">
        <h1 className="text-2xl font-bold text-white">
          <span className="text-accent-blue">Ask</span> the Scout
        </h1>
        <p className="text-text-secondary text-sm mt-1">
          AI-powered draft analyst — ask about GM tendencies, prospects, and team needs.
        </p>
        {draftContext && (
          <div className="mt-2 flex items-center gap-2 text-xs text-accent-blue/80 bg-accent-blue/5 border border-accent-blue/20 rounded-lg px-3 py-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-accent-blue/80 flex-shrink-0" />
            Draft context active — pick #{draftContext.current_pick}
            {draftContext.picking_team ? `, ${draftContext.picking_team} on the clock` : ''}
          </div>
        )}
      </div>

      {/* Chat area */}
      <div className="flex-1 overflow-y-auto min-h-0 pr-1">
        {isEmpty && (
          <div className="flex flex-col items-center justify-center h-full text-center pb-8">
            <div className="w-14 h-14 rounded-full bg-accent-blue/10 border border-accent-blue/30 flex items-center justify-center mb-4">
              <span className="text-2xl text-accent-blue font-black">S</span>
            </div>
            <p className="text-text-secondary text-sm mb-6 max-w-xs">
              Ask anything about the {DRAFT_YEAR} draft class, GM drafting patterns, or team positional needs.
            </p>
            <div className="flex flex-col gap-2 w-full max-w-sm">
              {SUGGESTED_QUESTIONS.map(q => (
                <button
                  key={q}
                  onClick={() => sendMessage(q)}
                  className="text-left px-4 py-2.5 rounded-lg bg-bg-card border border-border-subtle text-text-secondary hover:text-white hover:border-accent-blue/50 transition-colors text-sm"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        <AnimatePresence initial={false}>
          {messages.map((msg, i) => (
            <MessageBubble key={i} msg={msg} index={i} />
          ))}
        </AnimatePresence>

        {/* Tool status — shown while tool rounds execute, clears on first token */}
        {loading && toolStatus && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex justify-start mb-1 pl-9"
          >
            <span className="text-xs text-accent-blue/70 animate-pulse">{toolStatus}</span>
          </motion.div>
        )}

        {/* Typing indicator only shown before first token arrives */}
        {loading && messages[messages.length - 1]?.content === '' && !toolStatus && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex justify-start mb-3"
          >
            <div className="w-7 h-7 rounded-full bg-accent-blue/20 border border-accent-blue/40 flex items-center justify-center flex-shrink-0 mr-2 mt-0.5">
              <span className="text-xs text-accent-blue font-bold">S</span>
            </div>
            <div className="bg-bg-card border border-border-subtle rounded-xl rounded-bl-sm px-4 py-2.5">
              <div className="flex gap-1 items-center h-4">
                {[0, 1, 2].map(i => (
                  <span key={i} className="w-1.5 h-1.5 rounded-full bg-text-muted animate-bounce" style={{ animationDelay: `${i * 0.15}s` }} />
                ))}
              </div>
            </div>
          </motion.div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="flex-shrink-0 pt-3 border-t border-border-subtle">
        <div className="flex gap-2 items-end">
          <textarea
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about a GM, prospect, or team…"
            rows={1}
            className="flex-1 resize-none bg-bg-card border border-border-subtle rounded-xl px-4 py-2.5 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent-blue/60 transition-colors"
            style={{ minHeight: '2.75rem', maxHeight: '8rem' }}
            onInput={e => {
              const el = e.currentTarget
              el.style.height = 'auto'
              el.style.height = `${Math.min(el.scrollHeight, 128)}px`
            }}
          />
          <button
            onClick={() => sendMessage(input)}
            disabled={!input.trim() || loading}
            className="px-4 py-2.5 bg-accent-blue hover:bg-blue-500 disabled:opacity-40 text-white text-sm font-semibold rounded-xl transition-colors flex-shrink-0"
          >
            Send
          </button>
        </div>
        <p className="text-text-muted text-xs mt-2 text-center">
          Powered by Claude · Grounded in your draft database
        </p>
      </div>
    </main>
  )
}
