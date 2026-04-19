import { useState, useRef, useEffect, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { buildApiUrl } from '../../lib/api'
import { useDraftStore } from '../../stores/draftStore'
import { DRAFT_YEAR } from '../../lib/config'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

const SUGGESTED_QUESTIONS = [
  "Which teams need a center most?",
  "How does Kyle Dubas typically draft?",
  "Who are the top defensive prospects?",
  "What's Tampa Bay's drafting tendency?",
]

function MessageBubble({ msg }: { msg: Message }) {
  const isUser = msg.role === 'user'
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.15 }}
      className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-2`}
    >
      {!isUser && (
        <div className="w-6 h-6 rounded-full bg-accent-blue/20 border border-accent-blue/40 flex items-center justify-center flex-shrink-0 mr-1.5 mt-0.5">
          <span className="text-[10px] text-accent-blue font-bold">S</span>
        </div>
      )}
      <div
        className={`max-w-[85%] rounded-lg px-3 py-2 text-[13px] leading-relaxed whitespace-pre-wrap ${
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

export default function ScoutWidget() {
  const [open, setOpen] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [toolStatus, setToolStatus] = useState<string | null>(null)
  const [sessionId] = useState(() => crypto.randomUUID())
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const { lotteryResult, draftResult } = useDraftStore()

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
    if (open) bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading, open])

  async function sendMessage(text: string) {
    const trimmed = text.trim()
    if (!trimmed || loading) return

    setMessages(prev => [...prev, { role: 'user', content: trimmed }])
    setInput('')
    setLoading(true)
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
            if (payload.status) setToolStatus(payload.status)
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
    <>
      {/* Floating toggle button */}
      <button
        onClick={() => setOpen(o => !o)}
        className="fixed bottom-5 right-5 z-50 w-14 h-14 rounded-full bg-accent-blue hover:bg-blue-500 text-white shadow-lg shadow-accent-blue/30 flex items-center justify-center transition-colors"
        aria-label={open ? 'Close scout chat' : 'Open scout chat'}
      >
        {open ? (
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        ) : (
          <span className="text-lg font-black">S</span>
        )}
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            transition={{ duration: 0.18 }}
            className="fixed bottom-24 right-5 z-50 w-[min(22rem,calc(100vw-2.5rem))] h-[min(32rem,calc(100vh-8rem))] bg-bg-primary border border-border-default rounded-2xl shadow-2xl flex flex-col overflow-hidden"
          >
            {/* Header */}
            <div className="flex-shrink-0 px-4 py-3 border-b border-border-subtle bg-bg-secondary">
              <div className="flex items-center gap-2">
                <div className="w-7 h-7 rounded-full bg-accent-blue/20 border border-accent-blue/40 flex items-center justify-center">
                  <span className="text-xs text-accent-blue font-bold">S</span>
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-sm font-semibold text-white leading-tight">Ask the Scout</h3>
                  <p className="text-[11px] text-text-muted leading-tight">AI draft analyst</p>
                </div>
              </div>
              {draftContext && (
                <div className="mt-2 flex items-center gap-1.5 text-[10px] text-accent-blue/80 bg-accent-blue/5 border border-accent-blue/20 rounded px-2 py-1">
                  <span className="w-1 h-1 rounded-full bg-accent-blue/80 flex-shrink-0" />
                  Pick #{draftContext.current_pick}
                  {draftContext.picking_team ? ` · ${draftContext.picking_team}` : ''}
                </div>
              )}
            </div>

            {/* Chat area */}
            <div className="flex-1 overflow-y-auto min-h-0 px-3 py-3">
              {isEmpty && (
                <div className="flex flex-col items-center justify-center h-full text-center">
                  <p className="text-text-secondary text-xs mb-3 max-w-[16rem]">
                    Ask anything about the {DRAFT_YEAR} draft class.
                  </p>
                  <div className="flex flex-col gap-1.5 w-full">
                    {SUGGESTED_QUESTIONS.map(q => (
                      <button
                        key={q}
                        onClick={() => sendMessage(q)}
                        className="text-left px-3 py-2 rounded-lg bg-bg-card border border-border-subtle text-text-secondary hover:text-white hover:border-accent-blue/50 transition-colors text-xs"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              <AnimatePresence initial={false}>
                {messages.map((msg, i) => (
                  <MessageBubble key={i} msg={msg} />
                ))}
              </AnimatePresence>

              {loading && toolStatus && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex justify-start mb-1 pl-8">
                  <span className="text-[11px] text-accent-blue/70 animate-pulse">{toolStatus}</span>
                </motion.div>
              )}

              {loading && messages[messages.length - 1]?.content === '' && !toolStatus && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex justify-start mb-2">
                  <div className="w-6 h-6 rounded-full bg-accent-blue/20 border border-accent-blue/40 flex items-center justify-center flex-shrink-0 mr-1.5 mt-0.5">
                    <span className="text-[10px] text-accent-blue font-bold">S</span>
                  </div>
                  <div className="bg-bg-card border border-border-subtle rounded-lg rounded-bl-sm px-3 py-2">
                    <div className="flex gap-1 items-center h-3">
                      {[0, 1, 2].map(i => (
                        <span key={i} className="w-1 h-1 rounded-full bg-text-muted animate-bounce" style={{ animationDelay: `${i * 0.15}s` }} />
                      ))}
                    </div>
                  </div>
                </motion.div>
              )}
              <div ref={bottomRef} />
            </div>

            {/* Input */}
            <div className="flex-shrink-0 p-2 border-t border-border-subtle bg-bg-secondary">
              <div className="flex gap-1.5 items-end">
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask about a GM, prospect, or team…"
                  rows={1}
                  className="flex-1 resize-none bg-bg-card border border-border-subtle rounded-lg px-3 py-2 text-xs text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent-blue/60 transition-colors"
                  style={{ minHeight: '2.25rem', maxHeight: '6rem' }}
                  onInput={e => {
                    const el = e.currentTarget
                    el.style.height = 'auto'
                    el.style.height = `${Math.min(el.scrollHeight, 96)}px`
                  }}
                />
                <button
                  onClick={() => sendMessage(input)}
                  disabled={!input.trim() || loading}
                  className="px-3 py-2 bg-accent-blue hover:bg-blue-500 disabled:opacity-40 text-white text-xs font-semibold rounded-lg transition-colors flex-shrink-0"
                >
                  Send
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}
